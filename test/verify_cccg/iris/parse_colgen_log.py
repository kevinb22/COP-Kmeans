#!/usr/bin/python
# script to parse colgen logfiles and output a CSV

import sys
import os
import re
from datetime import datetime, timedelta

def getname(file):
    return os.path.splitext(os.path.split(file)[1])[0]

def parse_colgen_logfile(inputname):
    outdict = dict()
    outdict['name'] = getname(inputname)

    last_scip_line = None
    nSubp = 0
    totSubptime = timedelta(seconds=0)
    k = -1
    degens = 0

    for line in file(inputname, 'r'):
        if line.startswith('Cost of the initial clustering:'):
            outdict['inputqual'] = line[len('Cost of the initial clustering:'):].strip()
        if line.startswith('Solving Time'):
            outdict['time'] = line.split(':')[1].strip()
        if line.startswith('Solving Nodes'):
            outdict['nodes'] = line.split(':')[1].strip()
        if line.startswith('Running SiegBnB...'):
            nSubp += 1
            m = re.search(r"runtime: (.*)\n", line)
            if m != None:
                t = datetime.strptime(m.groups()[0].strip(), "%H:%M:%S.%f")
                totSubptime += timedelta(hours=t.hour, minutes=t.minute, seconds=t.second, microseconds=t.microsecond)
        if line.startswith('Primal Bound'):
            m = re.search(r"\(([0-9]+) solutions\)", line)
            if m != None:
                outdict['sols'] = m.groups()[0]
        if line.count('|') == 17:
            # last seen SCIP table-line
            last_scip_line = line
            tokens = line.split("|");
            if tokens[16].strip().replace('.','0').replace('+','0').replace('e','0').isdigit():
                outdict['bestPrimal'] = tokens[16].strip()
            if tokens[17].strip().replace('.','0').replace('+','0').replace('e','0').replace('%','0').isdigit():
                outdict['gap'] = tokens[17].strip().replace('%','0')
            
        if line.startswith('YES degen'):
            degens += 1
        if k != -1:
            # a bit ugly, start counting from 'objective value' till end
            # plus assume vars start with X_
            if line.startswith('X_'):
                k += 1
        if line.startswith('objective value:'):
            outdict['quality'] = line[len('objective value:'):].strip()
            k = 0
        if line.strip() == 'Solution verified.':
            outdict['SAT'] = 'OK'
        if line.startswith('number of clusters:'):
            outdict['nclusters'] = line[len('number of clusters:'):].strip()
        if line.startswith('number of datapoints:'):
            outdict['ndatapoints'] = line[len('number of datapoints:'):].strip()
        if line.startswith('number of constraints:'):
            outdict['nconstraints'] = line[len('number of constraints:'):].strip()
        if line.startswith('STABILIZED'):
            outdict['stabilized'] = 'YES'
        if line.startswith('FIND_'):
            outdict['all/best'] = line[len('FIND_'):].strip().lower()
        if line.startswith('branch and bound version:'):
            outdict['BnB'] = line[len('branch and bound version:'):].strip()
        if line.startswith('__TIMEOUT__:'):
            outdict['timeout'] = line[len('__TIMEOUT__:'):].strip()


    if last_scip_line == None:
        return outdict
    if not 'stabilized' in outdict:
        outdict['stabilized'] = 'NO'
    table = last_scip_line.split('|')
    outdict['mem'] = table[5].strip()
    outdict['vars'] = table[8].strip()
    outdict['cols'] = table[10].strip()
    outdict['degens'] = "%i"%degens
    outdict['k'] = "%i"%k
    outdict['nSubProbs'] = "%i"%nSubp
    outdict['totSubTime'] = "%f"%totSubptime.total_seconds()
    return outdict

# http://stackoverflow.com/questions/4836710/does-python-have-a-built-in-function-for-string-natural-sort
def natural_sort(l): 
    convert = lambda text: int(text) if text.isdigit() else text.lower() 
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ] 
    return sorted(l, key = alphanum_key)

if __name__ == "__main__":
    if len(sys.argv) <= 1:
        print "Usage: %s <log[1].txt> .. <log[n].txt>"%sys.argv[0]
        print "\tlog[i].txt\teach one is a colgen output for a certain setting"
        print "\t       \tFor example: '../bin/optclust -k 8 -d ../initData/Ruspini/Ruspini.raw > log.txt'"

    inputnames = sys.argv[1:]
    inputnames = natural_sort(inputnames)

    header = True
    legend = ['name', 'k', 'nclusters', 'ndatapoints', 'nconstraints', 'stabilized', 'all/best', 'BnB', 'quality', 'time', 'mem', 'nSubProbs', 'totSubTime', 'sols', 'cols', 'nodes', 'degens', 'inputqual', 'timeout', 'bestPrimal', 'gap', 'SAT']
    for inputname in inputnames:
        if header:
            print ";".join(legend)
            header = False

        outdict = parse_colgen_logfile(inputname)

        o = [outdict[x] if x in outdict else "-" for x in legend]
        print ";".join(o)



