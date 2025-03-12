import numpy as np

def pca(df, n):
    df_data = df.iloc[1:, :]
    df_data = df_data.to_numpy()
    mu = np.mean(df_data, axis=0)
    df_m = df_data - mu
    S = np.cov(df_m.T)
    eigen_val, eigen_vec = np.linalg.eig(S)
    sorted_index = np.flipud(np.argsort(eigen_val))
    sorted_eigen_val = eigen_val[sorted_index] # verifica che gli autovalori siano ordinati correttamente
    sorted_eigen_vec = eigen_vec[: , sorted_index]
    U = sorted_eigen_vec[: , 0:n]
    z_star = np.matmul(df_m, U) #proietta i dati nello spazio pc
    return z_star

def knn (x_train, y_train, x_test, k):
    distances = np.linalg.norm(x_train - x_test, axis=1)
    indices = np.argsort(distances)[:k]
    counts = np.bincount(y_train[indices].astype(int))
    return np.argmax(counts)

