import sys

def read(fileName):
    return dict([tuple(l.split(',')) for l in open(fileName).read().splitlines()[1:]])

def merge(fileNames):
    r = {}
    for f in fileNames:
        r.update(read(f))
    return r

def readScores(scoresFile='resources/scores.txt', t=0.9):
    files = []
    for line in open(scoresFile).read().splitlines():
        fields = line.split()
        fn,auc = fields[0], float(fields[3].split('=')[1])
        if auc < t: continue
        print fn,auc
        files.append(fn)
    return files

def write(d, f):
    items = [(int(k),v) for k,v in d.items()]
    items.sort(key=lambda t: t[0])
    f.write('\n'.join(['id,ACTION'] + [str(k)+','+str(v) for k,v in items])) 

if __name__ == '__main__':
    files = sys.argv[1:]
    d = merge(files)
    write(d, sys.stdout)
