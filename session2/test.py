from model import KMeans, Member, Cluster

kmeans = KMeans(20)
kmeans.load_data(r'data\tf_idf.txt')
kmeans.run(100, 'max_iters', 10)

print("Purity value: ", kmeans.compute_purity())
print("NMI value: ", kmeans.compute_NMI())