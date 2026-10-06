"""Ejercicio 3: generar transacciones y evolucionar un detector de fraude.

Instalación: python -m pip install deap numpy pandas scikit-learn
Ejecución:   python ejercicio_3_fraude_pg.py

Se generan datos SINTÉTICOS: las métricas no estiman el rendimiento con bancos
reales. Tres corridas de programación genética compiten en validación; la prueba
se consulta únicamente después de escoger el modelo. El CSV se guarda junto al
script, salvo que se indique otra ruta con --salida.
"""

from __future__ import annotations

import argparse
import operator
import random
from pathlib import Path

import numpy as np
import pandas as pd
from deap import base, creator, gp, tools
from sklearn.metrics import (
    balanced_accuracy_score, confusion_matrix, f1_score,
    precision_score, recall_score,
)
from sklearn.model_selection import train_test_split


SEMILLA_DATOS = 20260929
SEMILLA_PARTICIONES = 14
SEMILLAS_EVOLUCION = (121, 122, 123)
N = 5000
POBLACION = 240
GENERACIONES = 90
ALTURA_MAXIMA = 7


def generar_transacciones() -> pd.DataFrame:
    """Simula clases minoritarias y distribuciones solapadas, sin datos reales.

    Primero se asigna la clase latente y luego se muestrean atributos
    condicionados a ella. Un 1.2 % de etiquetas se invierte al final para
    representar incertidumbre, errores y factores no observados.
    """
    rng = np.random.default_rng(SEMILLA_DATOS)
    fraude = (rng.random(N) < 0.09).astype(int)
    ingreso = rng.lognormal(np.log(3_400_000), 0.55, N).clip(950_000, 16_000_000)
    hora = rng.integers(0, 24, N)
    nocturno = fraude.astype(bool) & (rng.random(N) < 0.48)
    hora[nocturno] = rng.integers(0, 6, nocturno.sum())

    monto = rng.lognormal(np.log(115_000), 1.05, N)
    monto[fraude == 1] *= rng.lognormal(np.log(5), 0.95, (fraude == 1).sum())
    monto = np.rint(np.clip(monto, 3_000, 15_000_000)).astype(int)

    distancia = rng.exponential(12, N)
    distancia[fraude == 1] = rng.exponential(80, (fraude == 1).sum())
    distancia = np.round(np.clip(distancia, 0, 600), 1)

    invertir = rng.random(N) < 0.012
    fraude[invertir] = 1 - fraude[invertir]
    return pd.DataFrame({
        "monto_cop": monto,
        "hora": hora,
        "distancia_km": distancia,
        "ingreso_mensual_cop": np.rint(ingreso).astype(int),
        "fraude": fraude,
    })


def preparar_datos(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Construye atributos y particiones estratificadas 60/20/20."""
    monto = df["monto_cop"].to_numpy()
    ingreso = df["ingreso_mensual_cop"].to_numpy()
    X = np.column_stack((
        monto / 1e6, df["hora"].to_numpy(), df["distancia_km"].to_numpy(),
        ingreso / 1e6, monto / ingreso,
    ))
    y = df["fraude"].to_numpy()
    entrenamiento, resto = train_test_split(
        np.arange(len(df)), test_size=0.4, random_state=SEMILLA_PARTICIONES, stratify=y,
    )
    validacion, prueba = train_test_split(
        resto, test_size=0.5, random_state=SEMILLA_PARTICIONES, stratify=y[resto],
    )
    return X, y, entrenamiento, validacion, prueba


def crear_conjunto_primitivas() -> gp.PrimitiveSetTyped:
    """Árboles tipados: operaciones reales internas y salida booleana."""
    conjunto = gp.PrimitiveSetTyped("FLAG", [float] * 5, bool)
    for i, nombre in enumerate(("m", "h", "d", "s", "r")):
        conjunto.renameArguments(**{f"ARG{i}": nombre})
    conjunto.addPrimitive(operator.gt, [float, float], bool, name="GT")
    conjunto.addPrimitive(operator.lt, [float, float], bool, name="LT")
    conjunto.addPrimitive(np.logical_and, [bool, bool], bool, name="AND")
    conjunto.addPrimitive(np.logical_or, [bool, bool], bool, name="OR")
    conjunto.addPrimitive(np.logical_not, [bool], bool, name="NOT")
    conjunto.addPrimitive(lambda s, u, v: np.where(s, u, v), [bool] * 3, bool, name="IF")
    conjunto.addPrimitive(operator.add, [float, float], float, name="ADD")
    conjunto.addPrimitive(operator.sub, [float, float], float, name="SUB")
    for valor in (0.01, 0.03, 0.05, 0.1, 0.2, 0.3, 0.5, 1., 2., 4., 6., 10., 20., 50., 100., 200.):
        conjunto.addTerminal(valor, float, name="C" + str(valor).replace(".", "_"))
    conjunto.addTerminal(True, bool, name="TRUE")
    conjunto.addTerminal(False, bool, name="FALSE")
    return conjunto


def crear_herramientas(conjunto: gp.PrimitiveSetTyped) -> base.Toolbox:
    if not hasattr(creator, "AptitudFraude"):
        creator.create("AptitudFraude", base.Fitness, weights=(1.0,))
    if not hasattr(creator, "ArbolFraude"):
        creator.create("ArbolFraude", gp.PrimitiveTree, fitness=creator.AptitudFraude)
    caja = base.Toolbox()
    caja.register("expresion", gp.genHalfAndHalf, pset=conjunto, min_=1, max_=3)
    caja.register("individuo", tools.initIterate, creator.ArbolFraude, caja.expresion)
    caja.register("poblacion", tools.initRepeat, list, caja.individuo)
    caja.register("seleccionar", tools.selTournament, tournsize=4)
    caja.register("cruzar", gp.cxOnePoint)
    caja.register("subarbol", gp.genFull, min_=0, max_=2)
    caja.register("mutar", gp.mutUniform, expr=caja.subarbol, pset=conjunto)
    caja.decorate("cruzar", gp.staticLimit(key=operator.attrgetter("height"), max_value=ALTURA_MAXIMA))
    caja.decorate("mutar", gp.staticLimit(key=operator.attrgetter("height"), max_value=ALTURA_MAXIMA))
    return caja


def predecir(arbol: gp.PrimitiveTree, conjunto: gp.PrimitiveSetTyped, X: np.ndarray) -> np.ndarray:
    """Ejecuta el programa; un árbol constante se expande a todas las filas."""
    salida = gp.compile(arbol, conjunto)(*X.T)
    return np.broadcast_to(np.asarray(salida, dtype=bool), len(X)).astype(int)


def evolucionar(
    conjunto: gp.PrimitiveSetTyped, caja: base.Toolbox,
    X: np.ndarray, y: np.ndarray, semilla: int,
) -> gp.PrimitiveTree:
    """Optimiza F1 en entrenamiento y penaliza ligeramente el tamaño."""
    random.seed(semilla)

    def aptitud(arbol: gp.PrimitiveTree) -> tuple[float]:
        try:
            estimado = predecir(arbol, conjunto, X)
        except (ArithmeticError, ValueError, TypeError):
            return (-100.0,)
        return (f1_score(y, estimado, zero_division=0) - 0.00035 * len(arbol),)

    poblacion = caja.poblacion(n=POBLACION)
    mejores = tools.HallOfFame(1)
    for generacion in range(GENERACIONES + 1):
        for individuo in poblacion:
            if not individuo.fitness.valid:
                individuo.fitness.values = aptitud(individuo)
        mejores.update(poblacion)
        if generacion == GENERACIONES:
            break
        siguiente = list(map(caja.clone, tools.selBest(poblacion, 2)))
        while len(siguiente) < POBLACION:
            padre, madre = map(caja.clone, caja.seleccionar(poblacion, 2))
            if random.random() < 0.78:
                padre, madre = caja.cruzar(padre, madre)
                del padre.fitness.values, madre.fitness.values
            if random.random() < 0.14:
                padre, = caja.mutar(padre)
                del padre.fitness.values
            if random.random() < 0.14:
                madre, = caja.mutar(madre)
                del madre.fitness.values
            siguiente.extend((padre, madre))
        poblacion = siguiente[:POBLACION]
    return mejores[0]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--salida", type=Path, default=Path(__file__).with_name("transacciones_sinteticas_pg.csv"),
                        help="Ruta del CSV generado")
    opciones = parser.parse_args()
    datos = generar_transacciones()
    opciones.salida.parent.mkdir(parents=True, exist_ok=True)
    datos.to_csv(opciones.salida, index=False)
    X, y, entrenamiento, validacion, prueba = preparar_datos(datos)
    print(f"Archivo: {opciones.salida}; filas: {len(datos)}; fraudes: {int(y.sum())}")
    print(f"Particiones: entrenamiento={len(entrenamiento)}, validación={len(validacion)}, prueba={len(prueba)}")

    conjunto = crear_conjunto_primitivas()
    caja = crear_herramientas(conjunto)
    candidatos = []
    for semilla in SEMILLAS_EVOLUCION:
        arbol = evolucionar(conjunto, caja, X[entrenamiento], y[entrenamiento], semilla)
        f1_entrenamiento = f1_score(y[entrenamiento], predecir(arbol, conjunto, X[entrenamiento]))
        f1_validacion = f1_score(y[validacion], predecir(arbol, conjunto, X[validacion]))
        candidatos.append((f1_validacion, -len(arbol), arbol, semilla, f1_entrenamiento))
        print(f"Semilla {semilla}: F1 entrenamiento={f1_entrenamiento:.4f}; "
              f"F1 validación={f1_validacion:.4f}; nodos={len(arbol)}", flush=True)

    # La selección se realiza ANTES de consultar las etiquetas de prueba.
    f1_validacion, _, mejor, semilla, f1_entrenamiento = max(candidatos, key=lambda c: c[:2])
    estimado = predecir(mejor, conjunto, X[prueba])
    tn, fp, fn, tp = confusion_matrix(y[prueba], estimado, labels=[0, 1]).ravel()
    print(f"Modelo elegido (semilla {semilla}): {mejor}")
    print(f"F1 entrenamiento={f1_entrenamiento:.4f}; F1 validación={f1_validacion:.4f}")
    print(f"Prueba: TN={tn}, FP={fp}, FN={fn}, TP={tp}")
    print(f"Precisión={precision_score(y[prueba], estimado):.4f}; "
          f"sensibilidad={recall_score(y[prueba], estimado):.4f}; "
          f"F1={f1_score(y[prueba], estimado):.4f}; "
          f"exactitud balanceada={balanced_accuracy_score(y[prueba], estimado):.4f}")


if __name__ == "__main__":
    main()
