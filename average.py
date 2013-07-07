import pandas as pd
import sys

def create_test_submission(prediction):
    content = ['id,ACTION']
    for i, p in enumerate(prediction):
        content.append('%i,%f' %(i+1,p))
    print '\n'.join(content)

if __name__ == '__main__':
    if len(sys.argv) < 2:
        print 'Not enough args'
        sys.exit(1)
    n = 1
    fn = sys.argv[1]
    #print fn
    p = pd.read_csv(fn).ACTION
    for fn in sys.argv[2:]:
        #print fn
        p += pd.read_csv(fn).ACTION
        n += 1
    create_test_submission(p / n)

