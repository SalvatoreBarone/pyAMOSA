"""
Copyright 2021-2023 Salvatore Barone <salvatore.barone@unina.it>

This is free software; you can redistribute it and/or modify it under
the terms of the GNU General Public License as published by the Free
Software Foundation; either version 3 of the License, or any later version.

This is distributed in the hope that it will be useful, but WITHOUT
ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
FITNESS FOR A PARTICULAR PURPOSE. See the GNU General Public License for
more details.

You should have received a copy of the GNU General Public License along with
RMEncoder; if not, write to the Free Software Foundation, Inc., 51 Franklin
Street, Fifth Floor, Boston, MA 02110-1301, USA.
"""
import json, os, sys, math
from itertools import islice
from distutils.dir_util import mkpath

class MultiFileCacheHandle:

    def __init__(self, directory : str, max_size_mb : int = 10):
        self.directory = directory
        self.max_size_mb = max_size_mb

    def read(self):
        cache = {}
        if os.path.isdir(self.directory):
            for f in os.listdir(self.directory):
                if f.endswith('.json'):
                    with open(f"{self.directory}/{f}") as j:
                        tmp = json.load(j)
                        cache = {**cache, **tmp}
        print(f"{len(cache)} cache entries loaded from {self.directory}")
        return cache

    def write(self, cache : list):
        if os.path.isdir(self.directory):
            for file in os.listdir(self.directory):
                if file.endswith('.json'):
                    os.remove(f"{self.directory}/{file}")
        else:
            mkpath(self.directory)
        total_entries = len(cache)
        total_size = sys.getsizeof(json.dumps(cache))
        avg_entry_size = math.ceil(total_size / total_entries)
        max_entries_per_file = int(self.max_size_mb * (2 ** 20) / avg_entry_size)
        splits = int(math.ceil(total_entries / max_entries_per_file))
        for item, count in zip(MultiFileCacheHandle.chunks(cache, max_entries_per_file), range(splits)):
            with open(f"{self.directory}/{count:09d}.json", 'w') as outfile:
                outfile.write(json.dumps(item))

    @staticmethod
    def chunks(data, max_entries : int):
        it = iter(data)
        for _ in range(0, len(data), max_entries):
            yield {k: data[k] for k in islice(it, max_entries)}