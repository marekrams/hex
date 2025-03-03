
class MapHex:

    def __init__(self, s2i, N):

        all_sites = sorted(ind for inds in s2i.values() for ind in inds)
        assert all_sites == list(range(N))

        assert min(site[0] for site in s2i) == min(site[1] for site in s2i) == 0
        Nx = max(site[0] for site in s2i) + 1
        Ny = max(site[1] for site in s2i) + 1
        assert all((x, y) in s2i for x in range(Nx) for y in range(Ny))

        self.dims = (Nx, Ny)
        self.s2i = s2i
        self.i2s = {ind: (site, loc) for site, inds in s2i.items() for loc, ind in enumerate(inds)}
        self.s2N = {site: len(inds) for site, inds in s2i.items()}

    def group_H(self, H, gen_op, gen_I):
        HH = {'local': [], 'nn': []}
        for inds, val in H.items():
            Hterm = {}
            for ind in inds:
                site, loc = self.i2s[ind]
                Nloc = self.s2N[site]
                Oloc = gen_op(loc, Nloc)
                if site in Hterm:
                    Hterm[site] = Oloc @ Hterm[site]
                else:
                    Hterm[site] = Oloc

            Hlist = [val]
            for site, op in sorted(Hterm.items()):
                Hlist.append(site)
                Hlist.append(op)
                II = gen_I(self.s2N[site])
                assert (op @ op - II).norm() < 1e-12, "op ** 2 should be I. "

            which = 'local' if len(Hterm) == 1 else 'nn'
            HH[which].append(Hlist)
        return HH


class MapHex127(MapHex):

    def __init__(self):
        s2i = {(0, 0): (0, 1, 2, 3, 14),      (0, 1): (4, 5, 6, 7, 15),        (0, 2): (8, 9, 10, 11, 16),        (0, 3): (12, 13, 17),
               (1, 0): (18, 19, 20, 21, 33),  (1, 1): (22, 23, 24, 25, 34),    (1, 2): (26, 27, 28, 29, 35),      (1, 3): (30, 31, 32, 36),
               (2, 0): (37, 38, 39, 40, 52),  (2, 1): (41, 42, 43, 44, 53),    (2, 2): (45, 46, 47, 48, 54),      (2, 3): (49, 50, 51, 55),
               (3, 0): (56, 57, 58, 59, 71),  (3, 1): (60, 61, 62, 63, 72),    (3, 2): (64, 65, 66, 67, 73),      (3, 3): (68, 69, 70, 74),
               (4, 0): (75, 76, 77, 78, 90),  (4, 1): (79, 80, 81, 82, 91),    (4, 2): (83, 84, 85, 86, 92),      (4, 3): (87, 88, 89, 93),
               (5, 0): (94, 95, 96, 97, 109), (5, 1): (98, 99, 100, 101, 110), (5, 2): (102, 103, 104, 105, 111), (5, 3): (106, 107, 108, 112),
               (6, 0): (113, 114, 115),       (6, 1): (116, 117, 118, 119),    (6, 2): (120, 121, 122, 123),      (6, 3): (124, 125, 126)}

        super().__init__(s2i=s2i, N=127)
