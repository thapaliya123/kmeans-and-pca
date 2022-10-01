<h1>Clustering</h1>
<ul>
    <li><code>Clustering</code> is the taks of dividing the population or data points into a number of groups such that data points in the same groups are more similar to other data points in the same group than those in other groups</li>
    <li>In other word, the aim is to segregate groups with similar traits and assign them into <code>clusters.</code></li>
    <li><code>Cluster</code> is a group of objects that are similar to other objects in the cluster and dissimilar to data points in other cluster.</li>
    <li><code>Usage:</code> <i><strong>Customer segmentation</strong> is one of the popular usage of clustering.</i></li>
    <li>
        <strong>Clustering Application</strong>
        <ol>
            <li>
                <code>Retail/Marketing</code>
                <ul>
                    <li>Identifying buying patterns of customers.</li>
                    <li>Recommending new books or movies to new customers.</li>
                </ul>
            </li>
            <li>
                <code>Banking</code>
                <ul>
                    <li>Fraud detection in credit card use.</li>
                    <li>Identifying clusters of customers(e.g. loyal customer)</li>
                </ul>
            </li>
            <li>
                <code>Insurance</code>
                <ul>
                    <li>Fraud detection in claims analysis.</li>
                </ul>
            </li>
            <li>
                <code>Publication Media</code>
                <ul>
                    <li>Auto-categorizing news based on their content.</li>
                    <li>Recommending similar news article.</li>
                </ul>
            </li>
            <li>
                <code>Medicine</code>
                <ul>
                    <li>Characterizing patient behavior.</li>
                </ul>
            </li>
        </ol>
    </li>
    <li>
        <strong>Different Clustering Algorithms</strong>
        <ol>
            <li>
                <code>Partitioned-based Clustering</code>
                <ul>
                    <li>Relatively Efficient</li>
                    <li>Used for medium and large sized databases.</li>
                    <li>Example: K-means, K-medians, Fuzzy c_Means</li>
                </ul>
            </li>
            <li>
                <code>Hierarchical Clustering</code>
                <ul>
                    <li>Produces trees of clusters.</li>
                    <li>Very intuitive i.e. easy to use and understand.</li>
                    <li>Generally use with small size datasets.</li>
                    <li>Example: Agglomerative, Divisive</li>
                </ul>
            </li>
            <li>
                <code>Density-based Clustering</code>
                <ul>
                    <li>Ability to produce arbitrary shaped clusters.</li>
                    <li>Useful, when there is noise in the dataset.</li>
                    <li>Example: DBSCAN</li>
                </ul>
            </li>
        </ol>
    </li>
</ul>

<h1>K-means Clustering</h1>
<ul>
    <li><code>K-means</code> is a type of partitioning clustering.</li>
    <li>It can group data only unsupervised based on the similarity of data points to each other.</li>
    <li>K-means divides the data into <code>k</code> non-overlapping subsets (cluster) without any cluster internal structure.</li>
    <li>Examples within a clusters are very similar.</li>
    <li>Examples across different cluster are very different.</li>
    <li>Generally, Dissimilarity metrics such as <code>Euclidean distance</code> is use to measure either data points are similar to each other or they are dissimilar.</li>
    <li>
        <strong>K-means Objective:</strong>
        <ol>
            <li><i>To form cluster in such a way that similar samples go into a cluster, and dissimilar samples fall into different clusters.</i></li>
            <li><i>To minimize the <strong>intra-cluster</strong> distances and maximize the <strong>inter-cluster</strong> distances.</i></li>
            <li><i>To divide data into non-overlapping clusters without any cluster-internal structure.</i></li>
        </ol>
    </li>
    <li>
        <strong>K-means Algorithm</strong>
        <ol>
            <li>
                <code>Start with randomly placing <strong>k</strong> centroids i.e. one for each cluster.</code>
                <ul>
                    <li>Farther centroid are better.</li>
                    <li>Randomly selected centroids can either be from datasets or random.</li>
                </ul>
            </li>
            <li>
                <code>Calculate the distance of each point from each centroid</code>
                <ul>
                    <li>Generally <strong>Euclidean distance</strong> is used.</li>
                    <li>Others distance metric such as Cosine similarity can also be used.</li>
                </ul>
            </li>
            <li>
                <code>Assign each data point (object) to its closest centroid, creating a cluster</code>
            </li>
            <li>
                <code>Recalculate the positions of the <strong>k</strong> centroids</code>
                <ul>
                    <li>New centroid are calculated by taking mean of all data points assosciated to each cluster.</li>
                </ul>
            </li>
            <li>
                <code>Repeat the steps 2-4, until the centroids no longer move</code>
                <ul>
                    <li>Meaning: Repeat process until two steps has a same centroid.</li>
                </ul>
            </li>
        </ol>
    </li>
    <li>
        <strong>K-means Accuracy</strong>
        <ol>
            <li>
                <code>External approach</code>
                <ul>
                    <li>Compare the clusters with the ground truth if it is available.</li>
                    <li>Usually don't have ground truth as K-means is unsupervised algorithm.</li>
                </ul>
            </li>
            <li>
                <code>Internal approach</code>
                <ul>
                    <li>Average the distance between data points and their centroids within a cluster.</li>
                </ul>
            </li>
        </ol>
    </li>
</ul>

<h1>Why Clustering?</h1>
<ul>
    <li>Exploratory data analysis</li>
    <li>Summary generation</li>
    <li>Outlier detection</li>
    <li>Finding duplicates in datasets</li>
    <li>Preprocessing steps for prediction or other tasks.</li>
</ul>

<h1>Principal Component Analysis(PCA)</h1>

- Principal Component Analysis is a Dimensionality Reduction technique that enables you to identify correlations and patterns in a dataset so that it can be transformed into a dataset of significantly fewer dimensions without loss of any important information.  

- **Need of PCA**       
    - A dataset with more number of features takes more time for training the model and make data processing and exploratory data analysis(EDA) more complex.

- **Steps involved in PCA**
    1. Standardize the data. (with mean=0 and variance=1).
    2. Compute the covariance matrix of dimensions.
    3. Obtain the Eigenvectors and Eigenvalues from the covariance matrix (can also be used correleation matrix or singular value decomposition).
    4. Sort eigenvalues in descending order and choose the top k Eigenvectors that correspond to the k larges Eigenvalues (k will become the number of dimensions of the new feature subspace k<=d, d is the number of original dimension).
    5. Construct the projection matrix W from the selected k Eigenvectors.
    6. Transform the original data set X via W to obtain the new k-dimensional feature subspace Y.

# References
1. https://cognitiveclass.ai/
2. https://medium.com/analytics-vidhya/dimensionality-reduction-using-principal-component-analysis-pca-41e364615766
3. https://towardsdatascience.com/a-complete-guide-to-principal-component-analysis-pca-in-machine-learning-664f34fc3e5a
4. https://www.youtube.com/watch?v=Bt4zfx2R9vA&ab_channel=ExploringtheMeaningOfMath
5. https://stattrek.com/matrix-algebra/covariance-matrix.aspx
