# 计算机算法

## 数据结构基础

* 数据结构的存储方式只有顺序存储（数组）和链式存储（链表）两种，其他数据结构都是基于这两种基本数据结构玩出的花样。

### 数组

* 「静态数组」就是一块连续的内存空间，我们可以通过索引来访问这块内存空间中的元素，这才是数组的原始形态。

* 「动态数组」是编程语言为了方便我们使用，在静态数组的基础上帮我们添加了一些常用的 API，比如 `push, insert, remove` 等等方法，这些 API 可以让我们更方便地操作数组元素。

* 静态数组在创建的时候就要确定数组的元素类型和元素数量。

* 数组连续内存的特性给了他随机访问能力，在 O(1) 的时间内直接获取到对应元素的值。

  > 因为我可以通过首地址和索引直接计算出目标元素的内存地址。计算机的内存寻址时间可以认为是 `O(1)`，所以数组的随机访问时间复杂度是 `O(1)`。

### 链表

* 链表的元素可以分散在内存空间的天涯海角，通过每个节点上的 `next, prev` 指针，将零散的内存块串联起来形成一个链式结构。
* 数组最大的优势是支持通过索引快速访问元素，而链表就不支持。

### 队列/栈

* 队列和栈都是「操作受限」的数据结构。说它操作受限，主要是和基本的数组和链表相比，它们提供的 API 是不完整的。
* 队列只能在一端插入元素，另一端删除元素；栈只能在某一端插入和删除元素。
* 队列是一种「先进先出」的数据结构，栈是一种「先进后出」的数据结构，就是这个道理。

### 哈希表

* 哈希表的底层实现就是一个数组（我们不妨称之为 `table`）。它先把这个 `key` 通过一个哈希函数（我们不妨称之为 `hash`）转化成数组里面的索引，然后增删查改操作和数组。

  * 只有哈希函数的复杂度是 `O(1)`，且合理解决哈希冲突的问题，才能保证增删查改的复杂度是 `O(1)`。

* 如果两个不同的 `key` 通过哈希函数得到了相同的索引，这种情况就叫做「哈希冲突」。因为 `hash` 函数相当于是把一个无穷大的空间映射到了一个有限的索引空间，所以必然会有不同的 `key` 映射到同一个索引上。

* 出现哈希冲突的情况有两种常见的解决方法，一种是**拉链法**，另一种是**线性探查法**（也经常被叫做**开放寻址法**）。

  * 拉链法：哈希表的底层数组并不直接存储 `value` 类型，而是存储一个链表，当有多个不同的 `key` 映射到了同一个索引上，这些 `key -> value` 对就存储在这个链表中。
  * 线性探查法：一个 `key` 发现算出来的 `index` 值已经被别的 `key` 占了，那么它就往后找，直到找到一个空的位置为止。

* 频繁出现哈希冲突的两个原因：

  1、哈希函数设计的不好，导致 `key` 的哈希值分布不均匀，很多 `key` 映射到了同一个索引上。

  2、哈希表里面已经装了太多的 `key-value` 对了，这种情况下即使哈希函数再完美，也没办法避免哈希冲突。

  > 前者属于数学范畴，后者是算法需要考虑的。因此引入“负载因子”概念。

* 负载因子是一个哈希表装满的程度的度量，一般来说，负载因子越大，说明哈希表里面的 `key-value` 对越多，哈希冲突的概率就越大。负载因子的计算公式也很简单，就是 `size / table.length`。其中 `size` 是哈希表里面的 `key-value` 对的数量，`table.length` 是哈希表底层数组的容量。当哈希表内元素达到负载因子时，哈希表会扩容。把哈希表底层 `table` 数组的容量扩大，把数据搬移到新的大数组中，这样负载因子就减小了。

  * 因为哈希表在达到负载因子时会扩容，这个扩容过程会导致哈希表底层的数组容量变化，哈希函数计算出来的索引也会变化，**所以哈希表的遍历顺序也会变化**。

### 二叉树

* 二叉树本身是比较简单的基础数据结构，但是很多复杂的数据结构都是基于二叉树的，比如红黑树（二叉搜索树）、二叉堆、图、字典树、并查集等等。

* 根节点、父节点、子节点、叶子节点。从根节点到最下方叶子节点经过的节点个数为二叉树的最大深度/高度。

* 二叉树类型：

  * 满二叉树：每一层节点都是满的
    * 节点个数很好算，$n = 2^{h}-1$
  * 完全二叉树(Perfect Binary Tree)：二叉树的每一层的节点都紧凑靠左排列，且除了最后一层，其他每层都必须是满的
    * 由于它的节点紧凑排列，如果从左到右从上到下对它的每个节点编号，那么父子节点的索引存在明显的规律
  * 二叉搜索树BST：对于树中的每个节点，其**左子树的每个节点**的值都要小于这个节点的值，**右子树的每个节点**的值都要大于这个节点的值。你可以简单记为「左小右大」。
    * 因为左小右大的特性，可以让我们在 BST 中快速找到某个节点，或者找到某个范围内的所有节点，这是 BST 的优势所在。

* 二叉树的递归-层序遍历：

  * BST 的中序遍历结构是有序的

  * 递归遍历（DFS）：

    * 二叉树的前/中/后序遍历，它们其实就是在二叉树遍历框架的不同位置写代码。前序位置的代码会在进入节点时执行；中序位置的代码会在左子树遍历完成后，遍历右子树之前执行；后序位置的代码会在左右子树遍历完成后执行。

    ```python
    class TreeNode:
        def __init__(self, val=0, left=None, right=None):
            self.val = val
            self.left = left
            self.right = right
    
    # 二叉树的遍历框架
    def traverse(root: TreeNode):
        if root is None:
            return
        // 前序位置
        traverse(root.left);
        // 中序位置
        traverse(root.right);
        // 后序位置
    ```

    

  <img src="./img/2.jpeg" alt="img" style="zoom:50%;" />

  * 层序遍历（BFS）：

    * 二叉树的层序遍历，顾名思义，就是一层一层地遍历二叉树。这个遍历方式需要借助队列来实现。

    ```python
    class TreeNode:
        def __init__(self, val=0, left=None, right=None):
            self.val = val
            self.left = left
            self.right = right
    
    def levelOrderTraverse(root: TreeNode):
        if root is None:
            return
        q = deque()
        q.append(root)
        while q:
            cur = q.popleft()
            # 访问 cur 节点
            print(cur.val)
    
            # 把 cur 的左右子节点加入队列
            if cur.left is not None:
                q.append(cur.left)
            if cur.right is not None:
                q.append(cur.right)
    
    def levelOrderTraverse(root):
        if root is None:
            return
        q = deque()
        q.append(root)
        # 记录当前遍历到的层数（根节点视为第 1 层）
        depth = 1
    
        while q:
            sz = len(q)
            for i in range(sz):
                cur = q.popleft()
                # 访问 cur 节点，同时知道它所在的层数
                print(f"depth = {depth}, val = {cur.val}")
    
                # 把 cur 的左右子节点加入队列
                if cur.left is not None:
                    q.append(cur.left)
                if cur.right is not None:
                    q.append(cur.right)
            depth += 1
    ```



### 图结构

* 图结构就是 多叉树结构的延伸。在树结构中，只允许父节点指向子节点，不存在子节点指向父节点的情况，子节点之间也不会互相链接；而图中没有那么多限制，节点之间可以相互指向，形成复杂的网络结构。

* 在无向图中，度就是每个节点相连的边的条数。由于有向图的边有方向，所以有向图中每个节点的度被细分为入度 (indegree)和出度（outdegree）。

* 邻接表和邻接矩阵是图结构的两种实现方法

  * 邻接表：把每个节点 x 的邻居都存到一个列表里，然后把 x 和这个列表映射起来，这样就可以通过一个节点 x 找到它的所有相邻节点。

  * 邻接矩阵：是一个二维布尔数组，我们权且称为 `matrix`，如果节点 `x` 和 `y` 是相连的，那么就把 `matrix[x][y]` 设为 `true`（上图中绿色的方格代表 `true`）。

  * 空间复杂度：对于一幅有 `V` 个节点，`E` 条边的图，邻接表的空间复杂度是 `O(V + E)`，而邻接矩阵的空间复杂度是 `O(V^2)`。

    > 所以如果一幅图的 E 远小于 $V^2$（稀疏图），那么邻接表会比邻接矩阵节省空间，反之，如果 E 接近 $V^2$（稠密图），二者就差不多了。

* 图的遍历就是多叉树遍历的延伸。主要的遍历方式还是深度优先搜索（DFS）和广度优先搜索（BFS）。唯一的区别是，树结构中不存在环，而图结构中可能存在环，所以我们需要标记遍历过的节点，避免遍历函数在环中死循环。

  * 图结构的深度优先搜索就是多叉树的递归遍历。具体来说，遍历图的所有「节点」时，需要 `visited` 数组在前序位置标记节点；遍历图的所有「路径」时，需要 `onPath` 数组在前序位置标记节点，在后序位置撤销标记。

    > 遍历节点的代码中，`visited` 数组的指责是保证每个节点只会被访问一次。**而对于图结构来说，要想遍历所有路径，可能会多次访问同一个节点，这是关键的区别**。

  * 图结构的广度优先搜索就是多叉树的层序遍历，只不过也需要一个 `visited` 数组来避免重复遍历节点。

    > 理论上 BFS 遍历也需要区分遍历所有「节点」和遍历所有「路径」，但是实际上 BFS 算法一般只用来求最短路径，不会用来求所有路径。你要求所有路径的话，去用 DFS 就行。如果只求最短路径的话，只需要遍历「节点」，因为按照 BFS 算法一层一层向四周扩散的逻辑，第一次遇到目标节点就是最短路径。

* 对于树结构，遍历所有「路径」和遍历所有「节点」是没什么区别的。对于图结构，遍历所有「路径」和遍历所有「节点」稍有不同。

  * 因为对于树结构来说，由于只能由父节点指向子节点，所以从根节点 `root` 出发，到任意一个节点 `targetNode` 的路径都是唯一的。换句话说，我遍历一遍所有节点之后，必然可以找到 `root` 到 `targetNode` 的路径，而且这个路径是唯一的。
  * 而对于图结构来说，由起点 `src` 到目标节点 `dest` 的路径可能不止一条。我们需要一个 `onPath` 数组，在进入节点时（前序位置）标记为正在访问，退出节点时（后序位置）撤销标记，这样才能遍历图中的所有路径，从而找到 `src` 到 `dest` 的所有路径。

### 二叉堆

* 二叉堆就是一种能够动态排序的数据结构。动态排序是指可以不断往数据结构里面添加或删除元素，数据结构会自动调整元素的位置，使得我们可以有序地从数据结构中读取元素。

  > 能动态排序的常用数据结构其实只有两个，一个是优先级队列（底层用二叉堆实现），另一个是二叉搜索树。二叉搜索树的用途更广泛，优先级队列能做的事情，二叉搜索树其实都能做。但优先级队列的 API 和代码实现相较于二叉搜索树更简单，所以一般能用优先级队列解决的问题，我们没必要用二叉搜索树。

  * 二叉堆是一种特殊的二叉树，这棵二叉树上的任意节点的值，都必须大于等于（或小于等于）其左右**子树所有节点**的值。如果是大于等于，我们称之为「大顶堆」，如果是小于等于，我们称之为「小顶堆」。

* 二叉堆的主要应用：

  * 数据结构优先级队列（Priority Queue）

    > 自动排序是有代价的，注意优先级队列 API 的时间复杂度，增删元素的复杂度是 `O(logN)`，其中 `N` 是当前二叉堆中的元素个数

  * 排序算法堆排序（Heap Sort）：把一个乱序的数组都 `push` 到一个二叉堆（优先级队列）里面，然后再一个个 `pop` 出来，就得到了一个有序的数组

* 二叉堆的主要操作：`sink`（下沉）和 `swim`（上浮），用以维护二叉堆的性质。二叉堆的难点在于 **你在插入或删除元素时，还要保持堆的性质**。**增删元素的复杂度是 `O(logN)`，其中 `N` 是当前二叉堆中的元素个数**。

  * 增：push 方法插入元素

    > 以小顶堆为例，向小顶堆中插入新元素遵循两个步骤：
    >
    > 1、先把新元素追加到二叉树底层的最右侧，保持完全二叉树的结构。此时该元素的父节点可能比它大，不满足小顶堆的性质。
    >
    > 2、为了恢复小顶堆的性质，需要将这个新元素不断上浮（`swim`），直到它的父节点比它小为止，或者到达根节点。此时整个二叉树就满足小顶堆的性质了。

  * 删：pop 方法删除元素

    > 以小顶堆为例，删除小顶堆的堆顶元素遵循两个步骤：
    >
    > 1、先把堆顶元素删除，把二叉树底层的最右侧元素摘除并移动到堆顶，保持完全二叉树的结构。此时堆顶元素可能比它的子节点大，不满足小顶堆的性质。
    >
    > 2、为了恢复小顶堆的性质，需要将这个新的堆顶元素不断下沉（`sink`），直到它的子节点比它小为止，或者到达叶子节点。此时整个二叉树就满足小顶堆的性质了。

  