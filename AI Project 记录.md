# AI Project 记录

<center>刘祺，你们的好爸爸

## 框架



## MXFold 2

训练脚本可以参考`train.py`

很独特的一点是prediction是在loss里面完成的，只有当`loss.item() > 0`的时候才`loss.backward()`，而且loss是`(pred - ref) / l`，贼奇怪

​	详见`loss.py::StructuredLoss::forward()`, `loss.py::StructuredLossWithTurner::forward()`

model的结构更复杂一些

`AbstractFold`中

​	`self.predict`为 `interface.predict_zuker`或者`interface.predict_mxfold`

​	`self.partfunc`为`interface.partfunc_zuker`或者`interface.partfunc_mxfold`

`AbstractFold` -> `ZukerFold`

`AbstractFold` -> `MixedFold`，`MixedFold`中有`self.zuker = ZukerFold()`和`self.turner = RNAFold()`

### Dataset

![image-20211123210327214](C:\Users\86191\AppData\Roaming\Typora\typora-user-images\image-20211123210327214.png)

有两种DataLoader：

`class FastaDataset(Dataset)`: 没看懂，而且几乎用不到（我从mxfold2 github上下的数据集全都是.bpseq格式的文件）

`class BPseqDataset(Dataset)`: 读取`.bpseq`格式的文件

#### BPSeq

例子：

![image-20211123210619195](C:\Users\86191\AppData\Roaming\Typora\typora-user-images\image-20211123210619195.png)

其中每一项为

`<idx, ch, idx_match>`: `ch` is the nucloid that positioned at `idx`, match with `idx_match`.

官方说明是

> *Basepair and sequence information is presented in a text-based format.*
>
> *Each row presents information for one nucleotide in the sequence;*
>
> *the first field is the position number, the second field is the nucleotide at that position,*
>
> *and the third field contains the position number of its basepair partner (or 0 when unpaired).*
>
> *All secondary and tertiary basepairs are represented in this file, but base triples are not.*

在这个DataLoader中，读取到的每一项**几乎**都是

`(filename, seq, torch.tensor(p))`

​	`filename`: 这个seq的来源文件名

​	`seq`: 这个seq的string表示

​	`p`: 这个seq的pairing information，`p[0]`一定为0（因为第0个核酸不存在，所以跟它配对的也不存在）（而且也因为不会出现`len(l)==4`的情况导致`structure_is_known = False`，我不知道这啥意思,`len(l)`按照官方文档来说一定等于3啊）

​		`p[idx] = idx_match`，表示第`i`个核酸与`idx_match`配对（如果`idx_match==0`迮表示不配对）

​		看`dataset.py::BPseqDataset::read()`的意思，似乎有可能`<idx, ch, idx_match>`中的`idx_match`会出现`'x.<>|'`之中的元素，很迷惑

### Loss





## E2EFold

训练脚本可以参考`e2e_learning_stage3.py`

### Dataset

![image-20211123205754779](C:\Users\86191\AppData\Roaming\Typora\typora-user-images\image-20211123205754779.png)

使用了ArchiveII和RNAStralign，

存放在pickle里面，`data_x, data_y, pairs, seq_length`

`class RNASSDataGenerator(object)` 数据读取和处理

​	`data_x`: sequence数据的encoding，长度固定为600

​	`seq`: 根据`data_x`解码而来，长度固定为600(因为演示的文件叫`all_600.pickle`)(使用`.`pad到600)

​	`data_y`: 意义不明，没有用到（`next_batch()`里面有用到，但是这个函数本身没有用）

​	`pairs`: pair配对数据，`pairs[i]`为一个`list`，里面的每一项都形如`[a, b]`，表示第a个核酸与第b个核酸配对；对称(有`[a, b]`必有`[b, a]`) **注意：**e2efold中，核酸编码从0开始（即，a, b可为0），与mxfold不一样！

​	`seq_length`: 顾名思义，sequence长度（真实长度）



`train_data` : `RNASSDataGenerator()`

​	（`for contacts, seq_embeddings, matrix_reps, seq_lens in train_generator:`）

​	`get_one_sample(index)`返回以`index`为序号的四元组

​		`contact`: `self.pairs2map(data_pair)`，`seq_len`$\times$`seq_len`($600\times 600$)的01-矩阵，假如第a个核酸与第b个核酸配对，则`contact[a, b]=1`，否则为0 

​		`data_seq`: 从`self.data_x`得来，sequence数据的encoding，长度固定为600

​		`matrix_rep`: `np.zeros(contact.shape)`，意义不明

​		`data_len`: 从`self.seq_length`得来，sequence长度（真实长度）

`train_set`: `Dataset(train_data)`, 

​	`__getitem__(index)` 返回`RNASSDataGenerator()`中的`get_one_sample(index)`

 	`__len__(self)` 返回`RNASSDataGenerator().len`（存疑）

`train_generator`: `torch.utils.data.DataLoader()`，拿来训练用

### Loss

很朴素的 `torch.nn.BCEWithLogitsLoss()`



## 总结

从根本上，两个model虽然数据集相似度贼高，但是model的输入不一样。要么就根据model的特点转化一下数据，要么就改model。





Modification:

​	e2efold的forward函数中加了一个Mask项：如果mask != None，就把输出mask一下

​	`ContactAttention_simple_fix_PE::__init__()`: 删去了没有用到的`device`参数

​	sry 改动有点多... 记不清了