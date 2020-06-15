from scipy.sparse import csr_matrix


def get_csr_matrix(df, rowname, colname, value=None, shape=None):
    row = df[rowname]
    col = df[colname]
    if value is None:
        value = [1]*len(row)

    return csr_matrix((value, (row,col)), shape=shape)
