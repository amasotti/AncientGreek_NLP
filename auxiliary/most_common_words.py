import collections
import pandas as pd
import matplotlib.pyplot as plt


def most_common(tokenized_file,n_most_common=50,verbose=False):
    w_count = {}
    for w in tokenized_file:
        if w in w_count.keys():
            w_count[w] += 1
        else:
            w_count[w] = 1

    w_count = collections.Counter(w_count)
    if verbose:
        for w, c in w_count.most_common(n_most_common):
            print(f"{w} : {c}",compact=True)
    return w_count

def create_mostcommon_df(tokenized_file,n_most_common=50,plot=False):
    w_count = most_common(tokenized_file=tokenized_file, n_most_common=n_most_common)
    df = pd.DataFrame(w_count.most_common(n_most_common), columns=['word', 'count'])
    if plot:
        df.plot(x='word',y='count')
        plt.show()
    return df
