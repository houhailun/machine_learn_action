# machine_learn_action
机器学习实战

ch2: KNN
K近邻算法实现
    伪代码：1、计算已知类别数据集中的每个点与新样本之间的距离
           2、按照距离递增排序
           3、选择与当前距离最小的K个点
           4、确定前k的点所在类别的频率
           5、返回前k个点频率最高的列别作为当前点的预测分类
NOTE: 特征必须归一化
优点:
    1、算法简单，便于理解
    2、既可以分类和回归
    3、对异常值不敏感
缺点: 
    1、样本不均衡
        问题：当某一个类的样本容量很大，而其他类样本数量很小时，很有可能导致当输入一个未知样本时，该样本的k个邻居中大数量类的样本占多数。
        但是这类样本并不接近目标样本，而数量小的这类样本靠近目标样本。这个时候，应该认为该未知样本属于数量小的样本所属的类。但是，
        knn之关系哪类样本的数量最多，而不考虑距离远近
        优化: 采用权值方法，针对距离小的邻居权值大，距离大的邻居权值小，这样可以避免一个类别的样本数量过大导致误判的情况
        权值设置：
        (1)反函数 weight = 1 / (distance + const)
        const的原因: 避免2个相同的样本出现distance=0的异常情况
        缺点: 为近邻分配很大的权重，稍微远一点的就会下降很快，有时候也会使算法对噪声数据变得更加敏感。
    2、计算量太大
        问题：(1)需要存储全部的训练数据  (2)计算量较大，因为每一个待分类样本都需要计算它与全体已知样本的距离，才能求得它的k个最近邻
        优化：kd树
        kd树(K-dimension tree)是一种对k维空间中的实例点进行存储以便对其进行快速检索的树形数据结构。kd树是是一种二叉树，表示对k维空间的一个划分，构造kd树相当于不断地用垂直于坐标轴的超平面将K维空间切分，构成一系列的K维超矩形区域。kd树的每个结点对应于一个k维超矩形区域。利用kd树可以省去对大部分数据点的搜索，从而减少搜索的计算量.
        构造平衡kd树算法： 
        （1）开始：构造根结点，根结点对应于包含T的k维空间的超矩形区域。选择为坐标轴，以T中所有实例的坐标的中位数为切分点，将根结点对应的超矩形区域切分为两个子区域。切分由通过切分点并与坐标轴垂直的超平面实现。由根结点生成深度为1的左、右子结点：左子结点对应坐标小于切分点的子区域，右子结点对应于坐标大于切分点的子区域。将落在切分超平面上的实例点保存在根结点。
        （2）重复。对深度为j的结点，选择为切分的坐标轴，，以该结点的区域中所有实例的坐标的中位数为切分点，将该结点对应的超矩形区域切分为两个子区域。切分由通过切分点并与坐标轴垂直的超平面实现。由该结点生成深度为j+1的左、右子结点：左子结点对应坐标小于切分点的子区域，右子结点对应坐标大于切分点的子区域。将落在切分超平面上的实例点保存在该结点
        搜索kd树: 利用kd树可以省去对大部分数据点的搜索，从而减少搜索的计算量。下面以搜索最近邻点为例加以叙述：给定一个目标点，搜索其最近邻，首先找到包含目标点的叶节点；然后从该叶结点出发，依次回退到父结点；不断查找与目标点最近邻的结点，当确定不可能存在更近的结点时终止。这样搜索就被限制在空间的局部区域上，效率大为提高。
　　  用kd树的最近邻搜索：　　
    （1）在kd树中找出包含目标点的叶结点：从根结点出发，递归的向下访问kd树。若目标点当前维的坐标值小于切分点的坐标值，则移动到左子结点，否则移动到右子结点。直到子结点为叶结点为止；
    （2）以此叶结点为“当前最近点”；
    （3）递归的向上回退，在每个结点进行以下操作：
　　  （a）如果该结点保存的实例点比当前最近点距目标点更近，则以该实例点为“当前最近点”；
　　  （b）当前最近点一定存在于该结点一个子结点对应的区域。检查该子结点的父结点的另一个子结点对应的区域是否有更近的点。具体的，检查另一个子结点对应的区域是否与以目标点为球心、以目标点与“当前最近点”间的距离为半径的超球体相交。如果相交，可能在另一个子结点对应的区域内存在距离目标更近的点，移动到另一个子结点。接着，递归的进行最近邻搜索。如果不相交，向上回退。
    （4）当回退到根结点时，搜索结束。最后的“当前最近点”即为的最近邻点。
重点：
    1、K值的选择
        太小会导致过拟合，很容易将一些噪声（如上图离五边形很近的黑色圆点）学习到模型中，而忽略了数据真实的分布！
        如果我们选取较大的k值，就相当于用较大邻域中的训练数据进行预测，这时与输入实例较远的（不相似）训练实例也会对预测起作用，使预测发生错误，k值的增大意味着整体模型变得简单。
    2、距离的选择：欧式距离、曼哈顿距离
    3、判别准则

ch3: 决策树

ch4: 贝叶斯

ch5: 逻辑回归

ch6: SVM

ch7: 集成学习

ch8: 线性回归

ch9: 待实现

ch10: Kmeans聚类

ch11: 关联规则
