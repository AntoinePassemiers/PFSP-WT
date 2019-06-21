# -*- coding: utf-8 -*-
# io.py: Read/write operations
# author : Antoine Passemiers

from pfspwt.instance import Instance


class PFSPWTIO:

    class PFSPException:
        """Read/write exception."""
        def __init__(self, *args, **kwargs):
            Exception.__init__(self, *args, **kwargs)

    @staticmethod
    def readline(f):
        """Reads next non-empty line of a given file.

        Parameters:
            f (:obj:`_io.TextIOWrapper`): Current file.

        Returns:
            str: Next non-empty line.
        """
        line = f.readline()
        while not (len(line) > 2 or line[0].isalnum()):
            line = f.readline()
            if line is None:
                raise PFSPWTIO.PFSPException('Reached EOF.')
        return line.replace('\n', '')

    @staticmethod
    def read(filepath):
        """Parses PFSP-WT instance from text file.

        Parameters:
            filepath (str): Location of the instance file.

        Returns:
            :obj:`pfspwt.Instance`: PFSP-WT problem instance.
        """
        with open(filepath, 'r') as f:
            el = PFSPWTIO.readline(f).split()
            n, m = int(el[0]), int(el[1])
            p = list()
            for i in range(n):
                el = PFSPWTIO.readline(f).split()
                p.append([int(el[i*2+1]) for i in range(m)])
            d, w = list(), list()
            if not f.readline()[:6] == 'Reldue':
                raise PFSPWTIO.PFSPException('Expected "Reldue" before deadlines.')
            for i in range(n):
                el = PFSPWTIO.readline(f).split()
                d.append(int(el[1]))
                w.append(int(el[3]))
        return Instance(p, d, w)
