import datetime
import sqlite3

import duckdb
import pandas

from basic_perm import FinitaryPermutation

timestamp = lambda: str(datetime.datetime.now())[:19]
log = lambda *z: print(timestamp(), "---", *z)


def factorial(n):
    z = 1
    for i in range(1, n + 1):
        z = z * i
    return z


def generate_fast(generators):
    if not generators:
        return {FinitaryPermutation.identity()}

    resG = {FinitaryPermutation.identity()}
    fringe = {FinitaryPermutation.identity()}  # Die "neuen" Elemente

    while fringe:
        new_elements = set()
        for g in generators:
            for r in fringe:
                candidate = g * r
                if candidate not in resG:
                    resG.add(candidate)
                    new_elements.add(candidate)
        fringe = new_elements  # Nur mit den neuen in die nächste Runde
    return sorted(set(sorted(list(resG))))


def generate_subgroup_of_sigma_infinity(generators=[]):
    if len(generators) == 0:
        return [FinitaryPermutation.identity()]
    if len(generators) > 0:
        G = [FinitaryPermutation.identity()] + generators
        G = sorted(set([g for g in G] + [~g for g in G]))
        last_lG, lG = 0, len(G)
        resG = [g for g in G]
        while last_lG < lG:
            for g in [gg for gg in G if gg != FinitaryPermutation.identity()]:
                resG += [g * r for r in resG]
            resG = sorted(set(resG))
            last_lG = lG
            lG = len(resG)
        return sorted(resG)


def An_generators(n: int = 0):
    if n in (0, 1, 2):  # trivial group
        g = []
    if n == 3:  # Z/3 on (0 1 2)
        g = [FinitaryPermutation.cycle_of_degree(3)]
    if n >= 4:
        if n % 2 == 0:
            g = [
                FinitaryPermutation.cycle_of_degree(3),
                FinitaryPermutation.identity(n=1)
                + FinitaryPermutation.cycle_of_degree(n - 1),
            ]
        else:
            g = [
                FinitaryPermutation.cycle_of_degree(3),
                FinitaryPermutation.cycle_of_degree(n),
            ]
    return g


def generate_brutal(generators):
    resG = {FinitaryPermutation.identity()}
    fringe = [FinitaryPermutation.identity()]  # Liste ist schneller beim Iterieren

    # Lokale Referenzen auf Methoden (Speedup!)
    resG_add = resG.add
    resG_contains = resG.__contains__

    # Wir binden die __mul__ Methoden der Generatoren vorab
    # Dann sparen wir uns den Dispatch-Check bei jedem Aufruf
    bound_muls = [g.__mul__ for g in generators]

    while fringe:
        new_fringe = []
        new_fringe_append = new_fringe.append

        for mul in bound_muls:
            for r in fringe:
                candidate = mul(r)
                if not resG_contains(candidate):
                    resG_add(candidate)
                    new_fringe_append(candidate)
        fringe = new_fringe
    rG = []
    leng = len(f"{len(resG)}")
    for i, g in enumerate(sorted(resG)):
        g.name = f"g_{i:0>{leng}} (order {g.order()})" if i > 0 else "e"
        rG += [g]
    return rG


def get_generating_set(all_group_elements=[]):
    if len(all_group_elements) == 0:
        return []
    candidates = sorted(
        [g for g in all_group_elements if g != FinitaryPermutation.identity()]
    )
    gen_set = [candidates[-1]]
    cur_gen = generate_brutal(gen_set)
    tot_gen = generate_brutal(all_group_elements)
    while len(cur_gen) < len(tot_gen):
        candidates = sorted([z for z in tot_gen if z not in cur_gen])
        c = candidates[-1]
        gen_set = gen_set + [c]
        cur_gen = generate_brutal(gen_set)
    better_gen = []
    for g in gen_set:
        if g < ~g:
            better_gen += [g]
        else:
            better_gen += [~g]
    leng = len(f"{len(better_gen)}")
    for i, g in enumerate(sorted(better_gen)):
        g.name = f"b_{i:0>{leng}}, order {g.order()}"
    return better_gen


def get_abstract_presentation(all_elements, generating_set):
    # generators: Liste deiner b_0, b_1...
    gens = generating_set
    identity = FinitaryPermutation.identity()

    # 1. Spanning Tree via BFS bauen
    # Mapping: Element -> Liste von Generator-Indizes (das kanonische Wort)
    canonical_words = {identity: []}
    queue = [identity]

    # Wir benutzen ein Set für schnellen Lookup beim Generieren des Baums
    visited = {identity}

    for g in queue:
        for idx, x in enumerate(gens):
            target = g * x
            if target not in visited:
                visited.add(target)
                # Der Pfad zum Ziel ist der Pfad zu g + der aktuelle Erzeuger
                canonical_words[target] = canonical_words[g] + [idx]
                queue.append(target)

    # 2. Relationen ernten (Fundamentalkreise schießen)
    abstract_relations = set()

    for g in all_elements:
        for idx, x in enumerate(gens):
            target = g * x

            # Das Wort, das wir durch die Kante (g -> target) erhalten:
            word_g = canonical_words[g]
            word_target = canonical_words[target]

            # Relation: Pfad(g) + x + Pfad(target)^-1
            # Invertieren des Ziel-Pfades: Indizes umdrehen und Vorzeichen wechseln
            inverse_target_word = [
                -i - 1 if i >= 0 else i for i in reversed(word_target)
            ]

            # Das rohe Wort im freien Monoid
            raw_relation = word_g + [idx] + inverse_target_word

            # Ein bisschen aufräumen: Triviale Stornos wie b_0 * b_0^-1 direkt streichen
            simplified_rel = simplify_abstract_word(raw_relation)

            if simplified_rel:  # Wenn es nicht die reine Identität ist
                # Wir packen es als Tupel in ein Set, um Duplikate zu filtern
                abstract_relations.add(tuple(simplified_rel))

    rels = universal_greedy_minimize(gens, abstract_relations)

    return gens, sorted(list(rels), key=len)


def universal_greedy_minimize(generators, raw_relations):
    identity = FinitaryPermutation.identity()

    # 1. Ordnungen holen
    orders = [gen.order() for gen in generators]
    torsion_relations = [tuple([idx] * orders[idx]) for idx in range(len(generators))]

    # 2. ÜBERSETZUNG: Negative Indizes in reine positive Potenzen umwandeln
    positive_relations = []
    for rel in raw_relations:
        pos_word = []
        for x in rel:
            if x < 0:
                # -1 -> Erzeuger 0. Seine Inversen-Potenz ist order - 1
                gen_idx = -x - 1
                pos_word.extend([gen_idx] * (orders[gen_idx] - 1))
            else:
                pos_word.append(x)
        positive_relations.append(tuple(pos_word))

    # 3. Zyklische Reduktion & Torsions-Filter (wie gehabt)
    candidates = []
    for rel in positive_relations:
        # Lösche triviale Torsions-Wiederholungen (z.B. 0,0,0) raus
        simplified = list(rel)
        changed = True
        while changed:
            changed = False
            for idx, order in enumerate(orders):
                pattern = [idx] * order
                # Einfaches Sublist-Stripping
                for i in range(len(simplified) - order + 1):
                    if simplified[i : i + order] == pattern:
                        del simplified[i : i + order]
                        changed = True
                        break

        if not simplified:
            continue

        # Zyklisch rotieren und das kleinste nehmen
        rotations = []
        for i in range(len(simplified)):
            rotations.append(tuple(simplified[i:] + simplified[:i]))
        if rotations:
            candidates.append(min(rotations))

    candidates = sorted(list(set(candidates)), key=len)

    # 4. Gierige Elimination (Jetzt klappt das Ersetzen!)
    minimal_interaction_set = list(candidates)
    i = 0
    while i < len(minimal_interaction_set):
        test_rel = minimal_interaction_set[i]
        current_working_system = torsion_relations + [
            r for r in minimal_interaction_set if r != test_rel
        ]

        # Nutzen der überarbeiteten check-Funktion
        if check_redundancy_via_group(test_rel, current_working_system, generators):
            minimal_interaction_set.remove(test_rel)
        else:
            i += 1

    return torsion_relations + minimal_interaction_set


def check_redundancy_via_group(target_rel, working_relations, generators):
    # Wenn wir nur positive Zahlen haben, ist das Invertieren eines Segments
    # im Kreis extrem trivial: Es ist der rückwärts gelesene Rest des Kreises!
    word = list(target_rel)

    changed = True
    while changed:
        changed = False
        for rel in working_relations:
            n = len(rel)
            # Wenn wir mehr als die Hälfte einer bekannten Relation sehen...
            for k in range(n // 2 + 1, n):
                for start in range(n):
                    rotated = rel[start:] + rel[:start]
                    segment = rotated[:k]
                    # Das Gegenstück ist der Rest des Kreises, aber rückwärts!
                    replacement = list(reversed(rotated[k:]))

                    # Suchen und Ersetzen im Wort
                    for idx in range(len(word) - len(segment) + 1):
                        if tuple(word[idx : idx + len(segment)]) == tuple(segment):
                            word[idx : idx + len(segment)] = replacement
                            # Direkt aufräumen, falls Torsionen entstehen
                            word = clean_pure_torsions(word, generators)
                            changed = True
                            break
                if changed:
                    break
            if changed:
                break

    return len(word) == 0


def clean_pure_torsions(word, generators):
    # Hilfsfunktion, um x^order direkt zu killen
    w = list(word)
    changed = True
    while changed:
        changed = False
        for idx, gen in enumerate(generators):
            pattern = [idx] * gen.order()
            for i in range(len(w) - len(pattern) + 1):
                if w[i : i + len(pattern)] == pattern:
                    del w[i : i + len(pattern)]
                    changed = True
                    break
    return w


def clean_and_minimize_relations(raw_relations):
    unique_minimal = set()

    for rel in raw_relations:
        # 1. Inverses auflösen (Da Ordnung 3: b_0^-1 ist b_0^2, b_1^-1 ist b_1^2)
        # -1 (Inverses von b_0) wird zu [0, 0]
        # -2 (Inverses von b_1) wird zu [1, 1]
        positive_word = []
        for x in rel:
            if x == -1:
                positive_word.extend([0, 0])
            elif x == -2:
                positive_word.extend([1, 1])
            else:
                positive_word.append(x)

        # 2. Triviale Potenzen kürzen (z.B. 0, 0, 0, 0 -> 0)
        # Da b^3 = id, können wir Dreiergruppen streichen
        simplified = []
        for x in positive_word:
            simplified.append(x)
            while (
                len(simplified) >= 3
                and simplified[-3] == simplified[-2] == simplified[-1]
            ):
                simplified.pop()
                simplified.pop()
                simplified.pop()

        if not simplified:
            continue

        # 3. Zyklische Reduktion (Bringe das Wort in seine kanonische Rotationsform)
        # Wenn wir [0, 1, 0, 1] rotieren, testen wir alle Starts und nehmen das kleinste
        rotations = []
        for i in range(len(simplified)):
            rot = simplified[i:] + simplified[:i]
            # Auch hier drinnen noch mal triviale Dreier wechkürzen, falls durch die Rotation entstanden
            clean_rot = []
            for r in rot:
                clean_rot.append(r)
                while (
                    len(clean_rot) >= 3
                    and clean_rot[-3] == clean_rot[-2] == clean_rot[-1]
                ):
                    clean_rot.pop()
                    clean_rot.pop()
                    clean_rot.pop()
            if clean_rot:
                rotations.append(tuple(clean_rot))

        if rotations:
            unique_minimal.add(min(rotations))

    return sorted(list(unique_minimal), key=len)


def simplify_abstract_word(word):
    """Kürzt triviale Nachbarschaften wie [0, -1] (b_0 * b_0^-1)"""
    # Da deine Erzeuger hier alle Ordnung 3 haben, ist b^-1 gleich b^2.
    # Aber um es allgemein zu halten, kürzen wir nur direkte Inversen-Nachbarn:
    stack = []
    for x in word:
        if stack and (stack[-1] == -x - 1 or x == -stack[-1] - 1):
            stack.pop()
        else:
            stack.append(x)
    return stack


def Sn_generators(n: int = 0):
    if n in (0, 1):  # trivial group
        g = []
    if n == 2:
        g = [FinitaryPermutation.cycle_of_degree(2)]
    if n == 3:  # Z/3 on (0 1 2)
        g = [
            FinitaryPermutation.cycle_of_degree(2),
            FinitaryPermutation.cycle_of_degree(3),
        ]
    if n >= 4:
        if n % 2 == 0:
            g = [
                FinitaryPermutation.cycle_of_degree(2),
                FinitaryPermutation.cycle_of_degree(3),
                FinitaryPermutation.identity(n=1)
                + FinitaryPermutation.cycle_of_degree(n - 1),
            ]
        else:
            g = [
                FinitaryPermutation.cycle_of_degree(2),
                FinitaryPermutation.cycle_of_degree(3),
                FinitaryPermutation.cycle_of_degree(n),
            ]
    return g


taus = FinitaryPermutation.transpositions_up_to()
cycles = FinitaryPermutation.cycles_up_to()

x = cycles[5]
y = taus[0] * taus[1]
assert x**5 == FinitaryPermutation.identity(), "x not a 5 cycle"
assert y**3 == FinitaryPermutation.identity(), "y not 3 torsion"
assert (x * y) ** 5 == FinitaryPermutation.identity(), "xy not 5 torsion"

## hmm, factorial(n) // 2 und len(An) passen iwie nicht zusammen
# for i in range(0, 32):
#     g = Sn_generators(i)
#     An = generate_brutal(g)
#     log(i, len(An))

# heisenberg gruppe in s9
# x = (012)(345)(678)
# y = (147)(285)

x = (
    FinitaryPermutation.cycle_of_degree(3)
    + FinitaryPermutation.cycle_of_degree(3)
    + FinitaryPermutation.cycle_of_degree(3)
)
x.name = "x"
y = FinitaryPermutation([(1, 4), (4, 7), (7, 1)]) * FinitaryPermutation(
    [(2, 8), (8, 5), (5, 2)]
)
y.name = "y"
c = (~FinitaryPermutation.cycle_of_degree(9)) ** 3
c.name = "c"
assert c == x * y * ~x * ~y

G = generate_brutal([x, y])
