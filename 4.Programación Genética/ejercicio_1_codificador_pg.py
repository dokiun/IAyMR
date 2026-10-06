"""Ejercicio 1: codificador BCD a siete segmentos por programación genética.

Instalación: python -m pip install deap numpy
Ejecución:   python ejercicio_1_codificador_pg.py

Se evolucionan siete árboles booleanos, uno por segmento. Si una salida no es
exacta, se corrigen sus entradas erróneas con minterminos booleanos. El programa
verifica al final las 112 salidas (16 entradas BCD x 7 segmentos).
"""

from __future__ import annotations

import operator
import random
from typing import Callable

import numpy as np
from deap import base, creator, gp, tools


# Orden de segmentos: a, b, c, d, e, f, g. Un 1 enciende el segmento.
# Los códigos BCD no válidos (10 a 15) apagan todo el visualizador.
PATRONES = (
    "1111110", "0110000", "1101101", "1111001", "0110011",
    "1011011", "1011111", "1110000", "1111111", "1111011",
) + ("0000000",) * 6
NOMBRES = "abcdefg"
POBLACION = 350
GENERACIONES = 135
ALTURA_MAXIMA = 7


def crear_conjunto_primitivas() -> gp.PrimitiveSet:
    """Define entradas, constantes y compuertas con salidas booleanas."""
    conjunto = gp.PrimitiveSet("BCD", 4)
    conjunto.renameArguments(ARG0="b3", ARG1="b2", ARG2="b1", ARG3="b0")
    conjunto.addPrimitive(np.logical_and, 2, name="AND")
    conjunto.addPrimitive(np.logical_or, 2, name="OR")
    conjunto.addPrimitive(np.logical_xor, 2, name="XOR")
    conjunto.addPrimitive(np.logical_not, 1, name="NOT")
    conjunto.addPrimitive(lambda s, u, v: np.where(s, u, v), 3, name="MUX")
    conjunto.addTerminal(True, name="ONE")
    conjunto.addTerminal(False, name="ZERO")
    return conjunto


def preparar_evolucion(conjunto: gp.PrimitiveSet) -> base.Toolbox:
    """Configura generación, selección, cruce y mutación de los árboles."""
    if not hasattr(creator, "AptitudSieteSegmentos"):
        creator.create("AptitudSieteSegmentos", base.Fitness, weights=(1.0,))
    if not hasattr(creator, "ArbolSieteSegmentos"):
        creator.create(
            "ArbolSieteSegmentos", gp.PrimitiveTree,
            fitness=creator.AptitudSieteSegmentos,
        )

    caja = base.Toolbox()
    caja.register("expresion", gp.genHalfAndHalf, pset=conjunto, min_=1, max_=3)
    caja.register("individuo", tools.initIterate, creator.ArbolSieteSegmentos, caja.expresion)
    caja.register("poblacion", tools.initRepeat, list, caja.individuo)
    caja.register("seleccionar", tools.selTournament, tournsize=4)
    caja.register("cruzar", gp.cxOnePoint)
    caja.register("subarbol", gp.genFull, min_=0, max_=2)
    caja.register("mutar", gp.mutUniform, expr=caja.subarbol, pset=conjunto)
    caja.decorate("cruzar", gp.staticLimit(key=operator.attrgetter("height"), max_value=ALTURA_MAXIMA))
    caja.decorate("mutar", gp.staticLimit(key=operator.attrgetter("height"), max_value=ALTURA_MAXIMA))
    return caja


def predecir(programa: Callable, entradas: np.ndarray) -> np.ndarray:
    """Evalúa incluso los árboles que devuelven una constante escalar."""
    salida = programa(*entradas.T)
    return np.broadcast_to(np.asarray(salida, dtype=bool), len(entradas)).copy()


def evolucionar_segmento(
    indice: int, entradas: np.ndarray, esperadas: np.ndarray,
    conjunto: gp.PrimitiveSet, caja: base.Toolbox,
) -> tuple[gp.PrimitiveTree, np.ndarray]:
    """Busca una función para un segmento y conserva el mejor árbol histórico."""
    random.seed(100 + indice)

    def aptitud(arbol: gp.PrimitiveTree) -> tuple[float]:
        try:
            resultado = predecir(gp.compile(arbol, conjunto), entradas)
        except (ArithmeticError, ValueError, TypeError):
            return (-1000.0,)
        errores = np.count_nonzero(resultado != esperadas[:, indice])
        return (1 - errores / 16 - 0.00005 * len(arbol),)

    poblacion = caja.poblacion(n=POBLACION)
    mejores = tools.HallOfFame(1)
    for generacion in range(GENERACIONES + 1):
        for individuo in poblacion:
            if not individuo.fitness.valid:
                individuo.fitness.values = aptitud(individuo)
        mejores.update(poblacion)
        if generacion == GENERACIONES:
            break

        # Elitismo: las dos mejores soluciones pasan directamente a la siguiente generación.
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

    mejor = mejores[0]
    return mejor, predecir(gp.compile(mejor, conjunto), entradas)


def corregir_con_minterminos(
    prediccion: np.ndarray, esperadas: np.ndarray,
) -> tuple[np.ndarray, list[tuple[int, int, int, int]]]:
    """Aplica una compuerta por cada caso erróneo de la tabla de verdad.

    Para una entrada i, el mintermino M_i es 1 exclusivamente en esa entrada.
    Si falta encender un segmento, se aplica OR M_i; si sobra, AND NOT M_i.
    """
    corregida = prediccion.copy()
    cambios = []
    for i, segmento in zip(*np.where(prediccion != esperadas)):
        anterior = int(corregida[i, segmento])
        nuevo = int(esperadas[i, segmento])
        mintermino = np.arange(16) == i
        if nuevo:
            corregida[:, segmento] |= mintermino
        else:
            corregida[:, segmento] &= ~mintermino
        cambios.append((int(i), int(segmento), anterior, nuevo))
    return corregida, cambios


def main() -> None:
    entradas = np.array(
        [[(i >> 3) & 1, (i >> 2) & 1, (i >> 1) & 1, i & 1] for i in range(16)],
        dtype=bool,
    )
    esperadas = np.array([[int(bit) for bit in patron] for patron in PATRONES], dtype=bool)
    conjunto = crear_conjunto_primitivas()
    caja = preparar_evolucion(conjunto)

    predicciones = []
    for segmento in range(7):
        arbol, salida = evolucionar_segmento(segmento, entradas, esperadas, conjunto, caja)
        predicciones.append(salida)
        print(f"Segmento {NOMBRES[segmento]}: {np.count_nonzero(salida != esperadas[:, segmento])} "
              f"errores; {len(arbol)} nodos; árbol = {arbol}", flush=True)

    original = np.column_stack(predicciones)
    corregida, cambios = corregir_con_minterminos(original, esperadas)
    print(f"Errores antes de corregir: {np.count_nonzero(original != esperadas)}")
    for i, segmento, anterior, nuevo in cambios:
        operacion = "OR M" if nuevo else "AND NOT M"
        print(f"Corrección {NOMBRES[segmento]}: entrada {i}, {anterior} -> {nuevo}; {operacion}_{i}")
    errores_finales = int(np.count_nonzero(corregida != esperadas))
    print(f"Verificación final: {corregida.size - errores_finales}/{corregida.size} salidas correctas")
    if errores_finales:
        raise AssertionError("El circuito final no reproduce la tabla de verdad")


if __name__ == "__main__":
    main()
