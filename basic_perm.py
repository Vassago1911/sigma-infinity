class FinitaryPermutation:
    def __init__(self, mapping=None, max_idx=None, name=""):
        # mapping: {source: target} nur für nicht-triviale Bewegungen
        self.map = dict(mapping) if mapping else {}
        # Wir brauchen den maximalen Index für Summe und Tensor
        if max_idx is not None:
            self.n = max_idx
        elif self.map:
            self.n = max(self.map.keys() | self.map.values()) + 1
        else:
            self.n = 0
        self.name = name

    def reduced(self):
        self.map = {k: self.map[k] for k in self.map.keys() if self.map[k] != k}
        return self

    def __call__(self, x):
        try:
            xx = int(x)
        except Exception as e:
            assert 1 == 0, "only integer inputs allowed"
            return
        assert xx == x, "only integer inputs allowed"
        assert x >= 0, "only non-negative integer inputs allowed"
        return self.map.get(x, x)

    def __invert__(self):
        # Wir vertauschen einfach Key und Value für alle Einträge im Mapping
        inv_map = {v: k for k, v in self.map.items()}

        # Der maximale Index (self.n) bleibt bei der Inversen gleich
        return FinitaryPermutation(
            mapping=inv_map,
            max_idx=self.n,
            name=f"({self.name})⁻¹" if self.name else "",
        )

    def __mul__(self, other):
        """Komposition: self * other (erst 'other', dann 'self')"""
        new_n = max(self.n, other.n)

        # Der relevante Bereich sind alle Indizes, die
        # in irgendeiner Weise von self oder other angefasst werden.
        # (Support union Range von beiden)
        relevant_indices = (
            set(self.map.keys())
            | set(self.map.values())
            | set(other.map.keys())
            | set(other.map.values())
        )

        new_map = {}
        for i in relevant_indices:
            # erst other(i), dann self(target_other)
            target = self(other(i))

            # Nur speichern, wenn es keine Identität ist
            if target != i:
                new_map[i] = target

        # Namenslogik
        s_name = self.name if self.name else "?"
        t_name = other.name if other.name else "?"

        return FinitaryPermutation(new_map, new_n, name=f"({s_name} o {t_name})")

    def __pow__(self, n):
        if n == -1:
            return ~self
        if n == 0:
            return self.identity()
        if n > 0:
            if n == 1:
                return self
            if n > 1:
                return self * (self ** (int(n - 1)))

    def __repr__(self):
        s = sorted(self.map.keys())
        t = [self(z) for z in s]
        l0 = list(map(str, s))
        l1 = list(map(str, t))
        max_digits = max(max(map(len, l0), default=0), max(map(len, l1), default=0))
        l0 = list(map(int, l0))
        l1 = list(map(int, l1))
        s_str = "|" + " ".join([f"{z: >{max_digits}}" for z in l0]) + "|"
        t_str = "|" + " ".join([f"{z: >{max_digits}}" for z in l1]) + "|"

        if self.name:
            total_str = f"{self.name}\n = "
        else:
            total_str = ""
        total_str = total_str + "" + s_str + "\n   " + t_str + "\n"
        return total_str

    def __add__(self, other):
        """Direkte Summe: self + other"""
        new_map = self.map.copy()
        # Verschiebe die Indizes von 'other' hinter den Bereich von 'self'
        shift = self.n
        for s, t in other.map.items():
            new_map[s + shift] = t + shift
        if self.name:
            s_name = f"{self.name}"
        else:
            s_name = "?"
        if other.name:
            t_name = f"{other.name}"
        else:
            t_name = "?"
        return FinitaryPermutation(
            new_map, self.n + other.n, name=f"( {s_name} (+) {t_name} )"
        )

    def __matmul__(self, other):
        """Tensorprodukt: self @ other (Kronecker-Style)"""
        # Indizes (i, j) werden gemappt auf i * other.n + j
        new_map = {}
        m = other.n
        for i in range(self.n):
            for j in range(other.n):
                src = i * m + j
                tgt = self(i) * m + other(j)
                if src != tgt:
                    new_map[src] = tgt
        if self.name:
            s_name = f"{self.name}"
        else:
            s_name = "?"
        if other.name:
            t_name = f"{other.name}"
        else:
            t_name = "?"
        return FinitaryPermutation(
            new_map, self.n * other.n, name=f"( {s_name} (x) {t_name} )"
        )

    def get_canonical_word(self):
        """Gibt die Indizes i der Transpositionen (i, i+1) zurück"""
        # Wir arbeiten auf einer Kopie der Liste der Werte
        current_list = [self(i) for i in range(self.n)]
        word = []

        # Klassischer Bubble-Sort-Mechanismus, der die Züge trackt
        # Dies erzeugt ein reduziertes Wort
        for i in range(len(current_list)):
            for j in range(len(current_list) - 1, i, -1):
                if current_list[j] < current_list[j - 1]:
                    # Vertausche in der Liste
                    current_list[j], current_list[j - 1] = (
                        current_list[j - 1],
                        current_list[j],
                    )
                    # Merke dir den Erzeuger s_{j-1}
                    word.append(j - 1)

        # Da wir 'rückwärts' zur Identität sortiert haben,
        # müssen wir die Sequenz umdrehen, um das Element zu bauen
        return word[::-1]

    def __hash__(self) -> int:
        return hash(tuple(self.get_canonical_word()))

    def order(self) -> int:
        if self == FinitaryPermutation.identity():
            return 0
        else:
            t = self
            i = 1
            while t != FinitaryPermutation.identity():
                t = t * self
                i += 1
            return i

    def __lt__(self, other):
        # 1. Bereinigte Maps holen
        map_a = self.reduced().map
        map_b = other.reduced().map

        # 2. Kriterium: Länge des Supports
        len_a = len(map_a)
        len_b = len(map_b)
        if len_a != len_b:
            return len_a < len_b

        # 3. Kriterium: Support sortiert vergleichen ("fängt früher an")
        support_a = tuple(sorted(map_a.keys()))
        support_b = tuple(sorted(map_b.keys()))
        if support_a != support_b:
            return support_a < support_b

        # 4. Kriterium: Werte parameterweise vergleichen (basierend auf sortierten Keys!)
        # Wir holen die Values in der exakt gleichen Reihenfolge wie die sortierten Keys
        range_a = tuple(map_a[k] for k in support_a)
        range_b = tuple(map_b[k] for k in support_b)

        return range_a < range_b

    def __eq__(self, other):
        try:
            return self.get_canonical_word() == other.get_canonical_word()
        except Exception as e:
            return False

    @classmethod
    def chi(cls, p, q):
        """Symmetrie-Isomorphismus für die direkte Summe: p + q -> q + p"""
        n, m = p.n, q.n
        new_map = {}
        # Die ersten n Indizes wandern um m nach hinten
        for i in range(n):
            new_map[i] = i + m
        # Die nächsten m Indizes wandern um n nach vorne
        for j in range(m):
            new_map[j + n] = j
        return cls(new_map, n + m, name=f"chi_{n, m}")

    @classmethod
    def sigma(cls, p, q):
        """Symmetrie-Isomorphismus für das Tensorprodukt: p @ q -> q @ p"""
        n, m = p.n, q.n
        new_map = {}
        # Mapping von (i * m + j) auf (j * n + i)
        for i in range(n):
            for j in range(m):
                src = i * m + j
                tgt = j * n + i
                if src != tgt:
                    new_map[src] = tgt
        return cls(new_map, n * m, name=f"sigma_{n, m}")

    @classmethod
    def dl(cls, p, q, r):
        """Linkes Distributivgesetz: p @ (q + r) -> (p @ q) + (p @ r)"""
        n, m, k = p.n, q.n, r.n
        new_map = {}
        # Der Index in p @ (q + r) ist i * (m + k) + j
        for i in range(n):
            # Fall 1: Der zweite Teil stammt aus q (j < m)
            for j in range(m):
                src = i * (m + k) + j
                tgt = i * m + j  # Liegt im ersten Block der Summe
                if src != tgt:
                    new_map[src] = tgt
            # Fall 2: Der zweite Teil stammt aus r (j >= m)
            for j in range(k):
                src = i * (m + k) + (j + m)
                tgt = (n * m) + (i * k + j)  # Liegt im zweiten Block (nach p @ q)
                if src != tgt:
                    new_map[src] = tgt
        return cls(new_map, n * (m + k), name=f"dl_{n, m, k}")

    @classmethod
    def dr(cls, p, q, r):
        """Rechtes Distributivgesetz: (p + q) @ r -> (p @ r) + (q @ r)"""
        n, m, k = p.n, q.n, r.n
        new_map = {}
        # Der Index in (p + q) @ r ist i * k + j
        # Wobei i der Index aus der Summe (p + q) ist.

        # Fall 1: i stammt aus p (i < n)
        for i in range(n):
            for j in range(k):
                src = i * k + j
                tgt = i * k + j  # Identität im ersten Block
                if src != tgt:
                    new_map[src] = tgt

        # Fall 2: i stammt aus q (i >= n)
        for i in range(m):
            for j in range(k):
                src = (i + n) * k + j
                tgt = (n * k) + (i * k + j)  # Offset um Größe von p @ r
                if src != tgt:
                    new_map[src] = tgt
        return cls(new_map, (n + m) * k, name=f"dr_{n, m, k}")

    @classmethod
    def identity(cls, n=0):
        if n == 0:
            return FinitaryPermutation(name="e")
        elif n > 0:
            return FinitaryPermutation(list(zip(range(n), range(n))), name=f"e_{n}")

    @classmethod
    def tau(cls, i):
        try:
            xx = int(i)
        except Exception as e:
            assert 1 == 0, "only non-negative integer i allowed"
            return
        assert xx == i, "only non-negative integer i allowed"
        assert i >= 0, "only non-negative integer i allowed"
        return FinitaryPermutation([(i, i + 1), (i + 1, i)], name=f"tau_{i}")

    @classmethod
    def cycle_of_degree(cls, n: int = 5):
        if n <= 1:
            return cls.identity()
        s = list(range(n))
        t = s[1:] + [s[0]]
        map = dict(zip(s, t))
        return cls(map, name=f"c_{n}")

    @classmethod
    def cycle_from_presentation(cls, t: tuple[int] = tuple([])):
        if len(t) == 0:
            return FinitaryPermutation.identity()
        else:
            l = list(t)
            r = l[1:] + [l[0]]
            m = list(zip(l, r))
            return FinitaryPermutation(m, name="c")

    @classmethod
    def cycles_up_to(cls, n: int = 32):
        return [cls.cycle_of_degree(k) for k in range(n)]

    @classmethod
    def transpositions_up_to(cls, n: int = 31):
        return [cls.tau(i) for i in range(n)]

    @classmethod
    def test_class(cls):
        FinitaryPermutation.check_bipermutative_axioms()

    @classmethod
    def check_bipermutative_axioms(cls):
        # + strictly monoidal
        e = FinitaryPermutation.identity()
        c2, c3, c4, c5, c6, c7 = (
            FinitaryPermutation.cycle_of_degree(2),
            FinitaryPermutation.cycle_of_degree(3),
            FinitaryPermutation.cycle_of_degree(4),
            FinitaryPermutation.cycle_of_degree(5),
            FinitaryPermutation.cycle_of_degree(6),
            FinitaryPermutation.cycle_of_degree(7),
        )
        assert e + c2 == c2, "implementation error, e not strictly + neutral"
        assert c2 + e == c2, "implementation error, e not strictly + neutral"
        assert (c2 + c3) + c5 == c2 + (c3 + c5), (
            "implementation error, + not strictly associative"
        )
        # @ strictly monoidal
        e0 = FinitaryPermutation([(0, 0)], name="e0")
        assert e0 @ c2 == c2, "implementation error, e0 not strictly @ neutral"
        assert c2 @ e0 == c2, "implementation error, e0 not strictly @ neutral"
        assert (c2 @ c3) @ c5 == c2 @ (c3 @ c5), (
            "implementation error, @ not strictly associative"
        )
        # +, @ symmetric monoidal
        s35 = FinitaryPermutation.chi(c3, c5)
        s53 = FinitaryPermutation.chi(c5, c3)
        c53 = c5 + c3
        c35 = c3 + c5
        assert s35 * s53 == e, "implementation error, chi not 'self-inverse'"
        assert s53 * s35 == e, "implementation error, chi not 'self-inverse'"
        assert s53 * c53 == c35 * s53, "implementation error, chi not the +-symmetry"
        assert s53 * c35 != c53 * s53, "implementation error, chi not the +-symmetry"
        assert s35 * c35 == c53 * s35, "implementation error, chi not the +-symmetry"
        assert s35 * c53 != c35 * s35, "implementation error, chi not the +-symmetry"
        s35 = FinitaryPermutation.sigma(c3, c5)
        s53 = FinitaryPermutation.sigma(c5, c3)
        c53 = c5 @ c3
        c35 = c3 @ c5
        assert s35 * s53 == e, "implementation error, sigma not 'self-inverse'"
        assert s53 * s35 == e, "implementation error, sigma not 'self-inverse'"
        assert s53 * c53 == c35 * s53, "implementation error, sigma not the @-symmetry"
        assert s53 * c35 != c53 * s53, "implementation error, sigma not the @-symmetry"
        assert s35 * c35 == c53 * s35, "implementation error, sigma not the @-symmetry"
        assert s35 * c53 != c35 * s35, "implementation error, sigma not the @-symmetry"
        # 567 -> 756 = 567 -> 576 -> 756
        s56_7 = FinitaryPermutation.chi(c5 + c6, c7)
        e5_s67 = FinitaryPermutation.identity(n=5) + FinitaryPermutation.chi(c6, c7)
        s57_e6 = FinitaryPermutation.chi(c5, c7) + FinitaryPermutation.identity(n=6)
        assert s56_7 == (s57_e6 * e5_s67), (
            "implementation error, +-symmetry chi not associative"
        )
        # 567 -> 675 = 567 -> 657 -> 675
        s5_67 = FinitaryPermutation.chi(c5, c6 + c7)
        s56_e7 = FinitaryPermutation.chi(c5, c6) + FinitaryPermutation.identity(n=7)
        e6_s57 = FinitaryPermutation.identity(n=6) + FinitaryPermutation.chi(c5, c7)
        assert s5_67 == e6_s57 * s56_e7, (
            "implementation error, +-symmetry chi not associative"
        )
        # 567 -> 756 = 567 -> 576 -> 756
        s56_7 = FinitaryPermutation.sigma(c5 @ c6, c7)
        e5_s67 = FinitaryPermutation.identity(n=5) @ FinitaryPermutation.sigma(c6, c7)
        s57_e6 = FinitaryPermutation.sigma(c5, c7) @ FinitaryPermutation.identity(n=6)
        assert s56_7 == (s57_e6 * e5_s67), (
            "implementation error, @-symmetry sigma not associative"
        )
        # 567 -> 675 = 567 -> 657 -> 675
        s5_67 = FinitaryPermutation.sigma(c5, c6 @ c7)
        s56_e7 = FinitaryPermutation.sigma(c5, c6) @ FinitaryPermutation.identity(n=7)
        e6_s57 = FinitaryPermutation.identity(n=6) @ FinitaryPermutation.sigma(c5, c7)
        assert s5_67 == e6_s57 * s56_e7, (
            "implementation error, @-symmetry sigma not associative"
        )
        # strict zero
        assert e @ c53 == e, "implementation error, e not strict zero"
        assert c53 @ e == e, "implementation error, e not strict zero"
        # distributors
        dl235 = FinitaryPermutation.dl(c2, c3, c5)
        dr235 = FinitaryPermutation.dr(c2, c3, c5)
        al = c2 @ (c3 + c5)
        ar = c2 @ c3 + c2 @ c5

        bl = (c2 + c3) @ c5
        br = c2 @ c5 + c3 @ c5
        assert dl235 * al == ar * dl235, "left distributivity not working"
        assert dr235 * bl == br * dr235, "right distributivity not working"
        assert dr235 * br == bl * dr235, "right distributivity not working"
        assert dl235 * ar != al * dl235, "left distributivity not working"
        # associative distributors
        # c7 @ ( c4 + c5 + c6 ) -> c7 @ c4 + c7 @ ( c5 + c6 ) -> c7 @ c4 + c7 @ c5 + c7 @ c6
        # = c7 @ ( c4 + c5 + c6 ) -> c7 @ ( c4 + c5 ) + c7 @ c6 -> c7 @ c4 + c7 @ c5 + c7 @ c6
        d7_4_56 = FinitaryPermutation.dl(c7, c4, c5 + c6)
        e74_d7_56 = FinitaryPermutation.identity(
            n=(c7 @ c4).n
        ) + FinitaryPermutation.dl(c7, c5, c6)
        d7_45_6 = FinitaryPermutation.dl(c7, c4 + c5, c6)
        d7_45_e76 = FinitaryPermutation.dl(c7, c4, c5) + FinitaryPermutation.identity(
            n=(c7 @ c6).n
        )
        assert e74_d7_56 * d7_4_56 == d7_45_e76 * d7_45_6, "dl not associative"
        # associative distributors
        # ( c4 + c5 + c6 ) @ c7 -> c4 @ c7 + ( c5 + c6 ) @ c7 -> c4 @ c7 + c5 @ c7 + c6 @ c7
        # = ( c4 + c5 + c6 ) @ c7 -> ( c4 + c5 ) @ c7 + c6 @ c7 -> c4 @ c7 + c5 @ c7 + c6 @ c7
        d4_56_7 = FinitaryPermutation.dr(c4, c5 + c6, c7)
        e47_d567 = FinitaryPermutation.identity(n=(c4 @ c7).n) + FinitaryPermutation.dr(
            c5, c6, c7
        )
        d45_6_7 = FinitaryPermutation.dr(c4 + c5, c6, c7)
        d457_e67 = FinitaryPermutation.dr(c4, c5, c7) + FinitaryPermutation.identity(
            n=(c6 @ c7).n
        )
        assert e47_d567 * d4_56_7 == d457_e67 * d45_6_7, "dr not associative"
        # additive symmetry of dl
        # 5 * ( 6 + 7 ) -> 5 * ( 7 + 6 ) -> 57 + 56 = 5 * ( 6 + 7 ) -> 56 + 57 -> 57 + 56
        e5_t67 = FinitaryPermutation.identity(n=5) @ FinitaryPermutation.chi(c6, c7)
        d576 = FinitaryPermutation.dl(c5, c7, c6)
        d567 = FinitaryPermutation.dl(c5, c6, c7)
        t57_56 = FinitaryPermutation.chi(c5 @ c6, c5 @ c7)
        assert t57_56 * d567 == d576 * e5_t67, "dl not +-symmetric"
        # additive symmetry of dr
        # ( 5 + 6 ) * 7 -> ( 6 + 5 ) * 7 -> 67 + 57 = ( 5 + 6 ) * 7 -> 57 + 67 -> 67 + 57
        t56_e7 = FinitaryPermutation.chi(c5, c6) @ FinitaryPermutation.identity(n=7)
        d657 = FinitaryPermutation.dr(c6, c5, c7)
        d567 = FinitaryPermutation.dr(c5, c6, c7)
        t67_57 = FinitaryPermutation.chi(c5 @ c7, c6 @ c7)
        assert t67_57 * d567 == d657 * t56_e7, "dr not +-symmetric"
        # @ associativity of dl
        # 4 @ ( 5 @ ( 6 + 7 )) -> 4 @ ( 56 + 57 ) -> 456 + 457
        # = 4 @ ( 5 @ ( 6 + 7 )) -> 45 @ ( 6 + 7 ) -> 456 + 457
        lhs = FinitaryPermutation.dl(c4, c5 @ c6, c5 @ c7) * (
            FinitaryPermutation.identity(n=4) @ FinitaryPermutation.dl(c5, c6, c7)
        )
        rhs = FinitaryPermutation.dl(c4 @ c5, c6, c7)
        assert lhs == rhs, "dl not @-associative"
        # @ associativity of dr
        # ( ( 4 + 5 ) @ 6 ) @ 7 -> ( 46 + 56 ) @ 7 -> 467 + 567
        # = ( ( 4 + 5 ) @ 6 ) @ 7 -> ( 4 + 5 ) @ 67 -> 467 + 567
        lhs = FinitaryPermutation.dr(c4 @ c6, c5 @ c6, c7) * (
            FinitaryPermutation.dr(c4, c5, c6) @ FinitaryPermutation.identity(n=7)
        )
        rhs = FinitaryPermutation.dr(c4, c5, c6 @ c7)
        assert lhs == rhs, "dr not @-associative"
        # middle @ associativity of dl-dr
        # 4 @ ( 5 + 6 ) @ 7 -> 4 @ ( 57 + 67 ) -> 457 + 467
        # = 4 @ ( 5 + 6 ) @ 7 -> ( 45 + 46 ) @ 7 -> 457 + 467
        lhs = FinitaryPermutation.dl(c4, c5 @ c7, c6 @ c7) * (
            FinitaryPermutation.identity(n=4) @ FinitaryPermutation.dr(c5, c6, c7)
        )
        rhs = FinitaryPermutation.dr(c4 @ c5, c4 @ c6, c7) * (
            FinitaryPermutation.dl(c4, c5, c6) @ FinitaryPermutation.identity(n=7)
        )
        assert lhs == rhs, "dl-dr not middle-@-associative"
        # mixed associativity of distributors
        # ( 4 + 5 ) @ ( 6 + 7 ) -> ( 4 + 5 ) @ 6 + ( 4 + 5 ) @ 7 -> 46 + 56 + 47 + 57
        # = ( 4 + 5 ) @ ( 6 + 7 ) -> 4 @ ( 6 + 7 ) + 5 @ ( 6 + 7 ) -> 46 + 47 + 56 + 57 -> 46 + 56 + 47 + 57
        lhs = (
            FinitaryPermutation.dr(c4, c5, c6) + FinitaryPermutation.dr(c4, c5, c7)
        ) * FinitaryPermutation.dl(c4 + c5, c6, c7)
        rhs = (
            (
                FinitaryPermutation.identity(n=(c4 @ c6).n)
                + FinitaryPermutation.chi(c4 @ c7, c5 @ c6)
                + FinitaryPermutation.identity(n=(c5 @ c7).n)
            )
            * (FinitaryPermutation.dl(c4, c6, c7) + FinitaryPermutation.dl(c5, c6, c7))
            * FinitaryPermutation.dr(c4, c5, c6 + c7)
        )
        assert lhs == rhs, "dl-dr not mixed associative for (a+b)(c+d)"
        # bipermutative
        # ( 4 + 5 ) * 6 -> 6 * ( 4 + 5 ) -> 64 + 65
        # = ( 4 + 5 ) * 6 -> 46 + 56 -> 64 + 65
        lhs = FinitaryPermutation.dl(c6, c4, c5) * FinitaryPermutation.sigma(
            c4 + c5, c6
        )
        rhs = (
            FinitaryPermutation.sigma(c4, c6) + FinitaryPermutation.sigma(c5, c6)
        ) * FinitaryPermutation.dr(c4, c5, c6)
        assert lhs == rhs, "dl-dr not @-symmetries of each other"


FinitaryPermutation.test_class()
