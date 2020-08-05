# Word2Vec
使用C++实现Word2Vec模型（计算词向量） 
Use C++ to implement Word2vec(word embeding tool)


项目是对Word2Vec的实现，本人参照peghoty的博客（https://www.cnblogs.com/peghoty/p/3857839.html）学习Word2Vec中的数学原理和代码实现。然后自己独立使用C++语言复现。

项目实现了基于Hierarchical Softmax 的 CBOW、Skip-gram模型以及基于Negative Sampling 的 CBOW模型。

语料库：使用基于维基百科的轻量级语料库文件，共有1000多万词语，暂时没有实现并行处理，单线程下需要一整天左右训练完成。

语料库已上传至百度网盘
链接：https://pan.baidu.com/s/1RbD1umuJX5mggU13Gz4nRw 
提取码：pjmk
