#include <iostream>
#include <algorithm>
#include <string>
#include <math.h>
#include <vector>
#include <queue>
#include <fstream>
#include <sstream>
#include <stdlib.h>
#define FILE_BUFFER_LENGTH 300000

//声明const常量
const int sig_table_size = 10000;//sigmoid数组的长度（切分细度）
const int vec_len = 100;//词向量的长度
//const int yuliao_len = 10000;//一行语料中词语的最大总数量
const int dic_max_len = 500000;//词典中的词语最大总数量
const int hash_size = 1000000;//hash数组的大小，比词典大小稍大，预留空位，用空间保证搜索速度
const char file_name[] = "11.txt";//语料文件
const int min_f = 0; //设置低频词门槛(出现次数)
const double max_fre = 1;//和高频词门槛(出现频率)
const int capacity = 1000000;//设置每批训练的单词量
const int range = 3;//以单词本身向上/下推3个单词，都算context
const int negative_sampling_length = 10000000;//负采样数组长度，1e8
const double learn_rate0 = 0.025;//初始学习率

//函数声明

//声明结构体

//词典中的词语
struct word {
	long long freq = 0;//记录词语的出现频率
	//std::vector<int> path;//记录从根节点到叶子节点的路径，path是一个整型vector数组,我觉得没必要记录，因为可以根据编码来推导
	std::string name = "";//记录单词字符串
	std::string code = "";//记录hufferman码
	double wv[vec_len] = {};
	double sita[vec_len] = {};
};

//Hufferman树的结点
struct node {
	node* left = NULL;//左节点
	node* right = NULL;//右节点
	double sita[vec_len] = {};//该节点对应的向量，只有非叶子节点的该向量才有意义
	std::string name = "";//该节点表示的单词，只有叶子节点才有
	std::string code = "";//记录每一个节点（除根节点）的hufferman代码，左1右0
	int fre = 0;//该节点及其子树的频率之和

	//比较node的大小，实现node按从大到小排序
	friend bool operator < (node a, node b)
	{
		return a.fre > b.fre;
	}
};

//比较word的大小，实现word按频率大小从小到大排序
struct CmpNode
{
	bool operator() (const node* a, const node* b)
	{
		return a->fre > b->fre;
	}
};


//声明全局变量
double sig_table[sig_table_size];//存储sigmoid结果，通过查表得到
//std::string yu_liao[yuliao_len];//存储语料中的一行词语，训练单位为一行
word dictionary[dic_max_len];//存储词典（跟语料的区别是词典中的词语是不重复的）
int vocab_hash[hash_size];//存储hash数组，即（词语的hash值，在dictionary中的位置）
int index_dic_now = 0;//记录dictionary的下一个空缺位置,可以表示第一次扫描时读入词典中的词语数
int dic_len;//表示词典再去除高频词、低频词后的长度
long long sum_word = 0;//记录读取的语料总词语数
std::priority_queue<node*, std::vector<node*>, CmpNode> nodes;    //nodes在建树过程中的节点存储其中
node* root = NULL;
std::queue<node*> oa;//用于遍历整个树时的队列(宽度搜索)
double learn_rate = 0.025;//学习率
std::string words_oneiter[capacity];//记录本批次的单词
std::string words_nextiter[capacity];//记录上一批次时某行遗留下的单词
int* negtive_sampling[10];//记录10个分负采样数组的头指针


//全局函数

//比较word的大小，实现word按频率大小从大到小排序
bool CmpWord(word a, word b) {
	return a.freq > b.freq;
}

//初始化Sigmoid表格，已测试
void InitSigTable() {
	for (int i = 0; i < sig_table_size; i++) {
		double realP = -6 + 12 / (double)sig_table_size * i;
		double eRealP = exp(realP);
		sig_table[i] = eRealP / (1 + eRealP);
	}
}

//通过查找Sigmoid表格来求值，已测试
double CalculateSigmoid(double x) {
	if (x >= 6)
		return 1;
	else if (x <= -6)
		return 0;
	else
		return sig_table[(int)floor((x + 6) / (12 / (double)sig_table_size))];
}

//转换词语的hash值
int GetHash(std::string word) {
	long long t = 0;
	for (int i = 0; i < word.length(); i++) {
		t += word[i] - 'a' + t * 26;
		t %= hash_size;
	}
	return t % hash_size;
}

//使用词语字符串查找其在字典中的下标
int GetIndexByWord(std::string word) {
	long long index = GetHash(word);
	while (true) {
		if (vocab_hash[index] == -1) {
			//该词语尚未纳入字典中
			return -1;
		}
		if (dictionary[vocab_hash[index]].name == word) {
			//匹配成功，返回该下标
			return index;
		}
		index++;//匹配失败，将index向下移动
		if (index >= hash_size) {
			index = 0;//如果超过了边界，就从头开始，但是实际情况应该很少会如此
		}
	}
}

//扫描到一个词语后，对词典进行更新
//分两种情况
//如果该单词已在词典中，则增加词典中词语的fre
//如果词典中尚未加入这个词语，就将其加入词典最后，并更新hashtable
void AddWord(std::string word) {
	int index_word = GetIndexByWord(word);
	if (index_word == -1) {
		//说明字典中尚无该单词，应向词典中加入改单词
		dictionary[index_dic_now].freq = 1;
		dictionary[index_dic_now].name = word;
		index_dic_now++;
		//并更新hash表
		int index = GetHash(word);
		while (vocab_hash[index] != -1) {
			index++;
			if (index >= hash_size) {
				index = 0;//如果超过了边界，就从头开始
			}
		}
		//经过上述过程后，将找到能够插入的位置index
		vocab_hash[index] = index_dic_now - 1;//这里要减一，因为之前加过一了
	}
	else {
		//字典中已有该单词，我们只需将这个词语的fre+1
		dictionary[vocab_hash[index_word]].freq++;
	}
	return;
}

//生成hufferman树
//返回值：树的根节点指针
void BuildHuff() {
	//node* root;//根节点
	//将词典中的词依次转换成node叶子节点，加入nodes中
	for (int i = 0; i < dic_len; i++) {
		node* leaf_node = new node();
		leaf_node->fre = dictionary[i].freq;
		leaf_node->name = dictionary[i].name;
		nodes.push(leaf_node);
	}
	//不断从vector<node>中选择最小的两项
	while (nodes.size() > 1) {
		node* right = nodes.top();//依次取出左右节点。按照左边大右边小的惯例（word2vec的作者是这么做的）
		nodes.pop();
		node* left = nodes.top();
		nodes.pop();
		//生成新的节点
		node* new_father = new node();
		new_father->fre = right->fre + left->fre;
		new_father->left = left;
		new_father->right = right;
		double* sita = new_father->sita;
		for (int i = 0; i < vec_len; i++) {
			sita[i] = ((double)rand() / RAND_MAX - 0.5) / vec_len;
		}
		//将新节点加入其中
		nodes.push(new_father);
	}
	root = nodes.top();
}

//生成w的负采样样本
int CreatNegSample(std::string w) {
	int index_rand, index_word;
	do {
		index_rand = (rand() % 100000000);
		index_word = negtive_sampling[index_rand / 10000000][index_rand % 10000000];
	} while (dictionary[index_word].name == w);
	return index_word;
}

//应用于Hierarchical Softmax的语料读取函数
//读取语料，并初始化词典D和语料数组C
//我搜集到的语料是xml文件，已将其处理成无标点符号的文本，每行表示一篇文章，函数的参数即预处理后的txt文件
//在这个函数中，需要完成以下任务：
//1.读取语料
//2.将词语放入字典
//3.将字典按照频率排序
//4.将字典中的高频词、低频词去除，并重新生成hash数组
//5.生成hufferman树
//6.遍历整个树
void ReadOneLineFromFile_H() {
	//1&2.读取语料
	std::string line;
	std::fstream fin(file_name);//创建一个fstream文件流对象
	while (std::getline(fin, line)) {//每次读取一行，最后碰到EOF会结束
		std::stringstream ss(line);//创建一个字符串流
		std::string buf;
		while (ss >> buf) {//将一行字符串，拆成一个单词形式的字符串
			AddWord(buf);//加入词语
			sum_word++;
		}
	}
	dic_len = index_dic_now;
	std::cout << ">>语料读取完毕" << std::endl;
	//3.将字典按照频率排序
	std::sort(dictionary, dictionary + index_dic_now, CmpWord);
	std::cout << ">>字典按频率重排序完毕" << std::endl;
	//4.将字典中的高频词、低频词去除，并重新生成hash数组
	//先以一定概率删除高频词
	int xi = 0, yi = 0;//分别表示遍历到的词典下表和已填充的新字典下表
	int max_f = max_fre * sum_word;//计算得到高频词门槛（出现次数）
	//max_f = 6000;//test，需要注释掉
	while (dictionary[xi].freq >= max_f) {
		int fre = dictionary[xi].freq;
		if ((double)rand() / RAND_MAX > ((double)max_f / fre + sqrt((double)max_f / fre))) {
			//应该舍弃
			xi++; dic_len--;
		}
		else {
			//该单词未舍弃
			if (xi != yi) {
				dictionary[yi].code = dictionary[xi].code;
				dictionary[yi].freq = dictionary[xi].freq;
				dictionary[yi].name = dictionary[xi].name;
			}
			xi++; yi++;
		}
	}
	//中间的单词全部都只需要向前移动即可
	while (yi < index_dic_now) {
		dictionary[yi].code = dictionary[xi].code;
		dictionary[yi].freq = dictionary[xi].freq;
		dictionary[yi].name = dictionary[xi].name;
		xi++; yi++;
	}
	//去除最后的低频词
	while (yi > 0 && dictionary[--yi].freq <= min_f) {
		dictionary[yi].code = "";
		dictionary[yi].freq = 0;
		dictionary[yi].name = "";
		dic_len--;
	}
	//重新生成hash数组
	memset(vocab_hash, -1, sizeof(int) * hash_size);//清空hash数组
	for (int i = 0; i < dic_len; i++) {
		std::string word = dictionary[i].name;
		int hash_temp = GetHash(word);//得到词语的hash值
		while (vocab_hash[hash_temp] != -1) {
			hash_temp++;
			if (hash_temp >= hash_size) {
				hash_temp = 0;//如果超过了边界，就从头开始
			}
		}
		vocab_hash[hash_temp] = i;
	}
	std::cout << ">>高低频词筛选完毕" << std::endl;
	//5.生成hufferman树
	BuildHuff();
	std::cout << ">>Hufferman树建立完毕" << std::endl;
	//6.遍历整个树，为叶子节点生成code，有一个问题，如果使用函数递归调用，可能会栈溢出，所以使用队列来遍历
	oa.push(root);
	while (!oa.empty()) {
		node* temp = oa.front();
		oa.pop();//出队
		if (temp->left != nullptr) {
			temp->left->code = temp->code + "1";
			temp->right->code = temp->code + "0";
			oa.push(temp->left);
			oa.push(temp->right);
		}
		else {
			//说明temp是叶子节点，接下来将该节点的code填到对应的word节点中
			int index_word = GetIndexByWord(temp->name);
			dictionary[vocab_hash[index_word]].code = temp->code;
		}
	}
	std::cout << ">>Hufferman树遍历完毕" << std::endl;
}

//用于负采样的语料读取方式
//读取语料，并初始化词典D和语料数组C
//我搜集到的语料是xml文件，已将其处理成无标点符号的文本，每行表示一篇文章，函数的参数即预处理后的txt文件
//在这个函数中，需要完成以下任务：
//1.读取语料
//2.将词语放入字典
//3.将字典按照频率排序
//4.将字典中的高频词、低频词去除，并重新生成hash数组
//5.初始化负采样数组
void ReadOneLineFromFile_N() {

	//1&2.读取语料
	std::string line;
	std::fstream fin(file_name);//创建一个fstream文件流对象
	while (std::getline(fin, line)) {//每次读取一行，最后碰到EOF会结束
		std::stringstream ss(line);//创建一个字符串流
		std::string buf;
		while (ss >> buf) {//将一行字符串，拆成一个单词形式的字符串
			AddWord(buf);//加入词语
			sum_word++;
		}
	}
	dic_len = index_dic_now;
	std::cout << ">>语料读取完毕" << std::endl;
	//3.将字典按照频率排序
	std::sort(dictionary, dictionary + index_dic_now, CmpWord);
	std::cout << ">>字典按频率重排序完毕" << std::endl;
	//4.将字典中的高频词、低频词去除，并重新生成hash数组
	//先以一定概率删除高频词
	int xi = 0, yi = 0;//分别表示遍历到的词典下表和已填充的新字典下表
	int max_f = max_fre * sum_word;//计算得到高频词门槛（出现次数）
	//max_f = 6000;//test，需要注释掉
	while (dictionary[xi].freq >= max_f) {
		int fre = dictionary[xi].freq;
		if ((double)rand() / RAND_MAX > ((double)max_f / fre + sqrt((double)max_f / fre))) {
			//应该舍弃
			xi++; dic_len--;
		}
		else {
			//该单词未舍弃
			if (xi != yi) {
				dictionary[yi].code = dictionary[xi].code;
				dictionary[yi].freq = dictionary[xi].freq;
				dictionary[yi].name = dictionary[xi].name;
			}
			xi++; yi++;
		}
	}
	//中间的单词全部都只需要向前移动即可
	while (yi < index_dic_now) {
		dictionary[yi].code = dictionary[xi].code;
		dictionary[yi].freq = dictionary[xi].freq;
		dictionary[yi].name = dictionary[xi].name;
		xi++; yi++;
	}
	//去除最后的低频词
	while (yi > 0 && dictionary[--yi].freq <= min_f) {
		dictionary[yi].code = "";
		dictionary[yi].freq = 0;
		dictionary[yi].name = "";
		dic_len--;
	}
	//重新生成hash数组
	memset(vocab_hash, -1, sizeof(int) * hash_size);//清空hash数组
	for (int i = 0; i < dic_len; i++) {
		std::string word = dictionary[i].name;
		int hash_temp = GetHash(word);//得到词语的hash值
		while (vocab_hash[hash_temp] != -1) {
			hash_temp++;
			if (hash_temp >= hash_size) {
				hash_temp = 0;//如果超过了边界，就从头开始
			}
		}
		vocab_hash[hash_temp] = i;
	}
	std::cout << ">>高低频词筛选完毕" << std::endl;
	
	//初始化负采样数组
	for (int i = 0; i < 10; i++) {
		int* negative_sampling_temp = new int[negative_sampling_length];
		negtive_sampling[i] = negative_sampling_temp;
	}
	long long total_num = 0;//筛选后的所有词语数
	double len_now = 0;//目前已读取的长度
	for (int i = 0; i < dic_len; i++) {
		total_num += dictionary[i].freq;//计算筛选后的所有词语数
		for (int j = 0; j < vec_len; j++) {
			dictionary[i].sita[j] = ((double)rand() / RAND_MAX - 0.5) / vec_len;//顺便初始化负采样中的单词sita向量 
		}
	}
	for (int i = 0; i < dic_len; i++) {
		double len = (double)dictionary[i].freq / total_num;
		for (int j = ceil(len_now / 0.00000001); j <= floor((len_now + len) / 0.00000001); j++) {
			negtive_sampling[j / 10000000][(j - 1) % 10000000] = i;
		}
	}
}

//训练skip-gram_H
void trainSKIPGRAM_H() {
	std::string line;
	int read_now = 0;//记录该批次已读单词数量
	//分批次读入并训练
	std::fstream fin(file_name);//创建一个fstream文件流对象
	long long num = 0;//已训练的样本数
	while (std::getline(fin, line)) {//每次读取一行，最后碰到EOF会结束
		std::stringstream ss(line);//创建一个字符串流
		std::string buf;
		while (ss >> buf) {//将一行字符串，拆成一个单词形式的字符串
			words_oneiter[read_now++] = buf;//放入该批次的词槽中
			if (read_now < capacity) {
				continue;//跳过后面的训练，继续填充词槽
			}
			learn_rate = learn_rate0 * (1 - (double)num / (sum_word + 1));
			if (learn_rate < learn_rate0 * 0.0001)learn_rate = learn_rate0 * 0.0001;
			//开始训练,每个单词相当于一个样本
			num += capacity;
			std::cout << "<<进度：" << num << std::endl;
			for (int j = 0; j < read_now; j++) {
				double* Xw = dictionary[vocab_hash[GetIndexByWord(words_oneiter[j])]].wv;
				for (int k = j - range; k <= j + range; k++) {//对每一个周边单词遍历
					if (k == j || k < 0 || k >= read_now)
						continue;
					double e[vec_len] = {};//声明e
					word u = dictionary[vocab_hash[GetIndexByWord(words_oneiter[k])]];//Context中的单词
					node* temp = root;//temp表示在逐节点下降过程中的当前节点
					for (int p = 0; p < u.code.length() - 1; p++) {
						double XwSita = 0;
						double* sita = temp->sita;
						for (int q = 0; q < vec_len; q++) {
							XwSita += sita[q] * Xw[q];
						}
						//计算q
						double q = CalculateSigmoid(XwSita);
						//计算g
						double dju = 0;
						if (u.code[p] == '1')
							dju = 1;
						double g = learn_rate * (1 - dju - q);
						//更新e
						for (int q = 0; q < vec_len; q++) {
							e[q] += g * sita[q];
						}
						//更细sita
						for (int q = 0; q < vec_len; q++) {
							sita[q] += g * Xw[q];
						}
						//递归到下一个节点
						if (dju)
							temp = temp->left;
						else
							temp = temp->right;
					}
					//更新v(w)
					for (int p = 0; p < vec_len; p++) {
						Xw[p] += e[p];
					}
				}
			}
			read_now = 0;
		}
	}
}

//训练CBOW_H
void trainCBOW_H() {
	std::string line;
	int read_now = 0;//记录该批次已读单词数量
	//分批次读入并训练
	std::fstream fin(file_name);//创建一个fstream文件流对象
	long long num = 0;//记录以训练样本数
	while (std::getline(fin, line)) {//每次读取一行，最后碰到EOF会结束
		std::stringstream ss(line);//创建一个字符串流
		std::string buf;
		while (ss >> buf) {//将一行字符串，拆成一个单词形式的字符串
			words_oneiter[read_now++] = buf;//放入该批次的词槽中
			if (read_now < capacity) {
				continue;//跳过后面的训练，继续填充词槽
			}
			learn_rate = learn_rate0 * (1 - (double)num / (sum_word + 1));
			if (learn_rate < learn_rate0 * 0.0001)learn_rate = learn_rate0 * 0.0001;
			//开始训练,每个单词相当于一个样本
			num += capacity;
			std::cout << "<<进度：" << num << std::endl;
			for (int j = 0; j < read_now; j++) {
				//计算 Context向量和
				double e[vec_len] = {}; double Xw[vec_len] = {};
				for (int k = 1; k <= range; k++) {
					if (j - k >= 0) {
						for (int p = 0; p < vec_len; p++) {
							Xw[p] += dictionary[vocab_hash[GetIndexByWord(words_oneiter[j - k])]].wv[p];
						}
					}
					if (j + k < read_now) {
						for (int p = 0; p < vec_len; p++) {
							Xw[p] += dictionary[vocab_hash[GetIndexByWord(words_oneiter[j + k])]].wv[p];
						}
					}
				}
				//使用梯度训练
				std::string code = dictionary[vocab_hash[GetIndexByWord(words_oneiter[j])]].code;
				node* temp = root;//temp表示在逐节点下降过程中的当前节点
				for (int k = 0; k < code.length() - 1; k++) {//对经过的节点进行遍历
					double XwSita = 0;
					double* sita = temp->sita;
					for (int p = 0; p < vec_len; p++) {
						XwSita += sita[p] * Xw[p];
					}
					//计算q
					double q = CalculateSigmoid(XwSita);
					//计算g
					double djw = 0;
					if (code[k] == '1')
						djw = 1;
					double g = learn_rate * (1 - djw - q);
					//更新e
					for (int p = 0; p < vec_len; p++) {
						e[p] += g * sita[p];
					}
					//更细sita
					for (int p = 0; p < vec_len; p++) {
						sita[p] += g * Xw[p];
					}
					//递归到下一个节点
					if (djw)
						temp = temp->left;
					else
						temp = temp->right;
				}
				//更新周边词语的词向量
				for (int k = 1; k <= range; k++) {
					if (j - k >= 0) {
						double* temp = dictionary[vocab_hash[GetIndexByWord(words_oneiter[j - k])]].wv;
						for (int p = 0; p < vec_len; p++) {
							temp[p] += e[p];
						}
					}
					if (j + k < read_now) {
						double* temp = dictionary[vocab_hash[GetIndexByWord(words_oneiter[j + k])]].wv;
						for (int p = 0; p < vec_len; p++) {
							temp[p] += e[p];
						}
					}
				}
			}
			read_now = 0;
		}
	}
}


//训练CBOW_N
void trainCBOW_N() {
	std::string line;
	int read_now = 0;//记录该批次已读单词数量
	//分批次读入并训练
	std::fstream fin(file_name);//创建一个fstream文件流对象
	long long num = 0;//记录以训练样本数
	while (std::getline(fin, line)) {//每次读取一行，最后碰到EOF会结束
		std::stringstream ss(line);//创建一个字符串流
		std::string buf;
		while (ss >> buf) {//将一行字符串，拆成一个单词形式的字符串
			words_oneiter[read_now++] = buf;//放入该批次的词槽中
			if (read_now < capacity) {
				continue;//跳过后面的训练，继续填充词槽
			}
			learn_rate = learn_rate0 * (1 - (double)num / (sum_word + 1));
			if (learn_rate < learn_rate0 * 0.0001)learn_rate = learn_rate0 * 0.0001;
			//开始训练,每个单词相当于一个样本
			num += capacity;
			std::cout << "<<进度：" << num << std::endl;
			for (int j = 0; j < read_now; j++) {
				//计算 Context向量和
				double e[vec_len] = {}; double Xw[vec_len] = {};
				for (int k = 1; k <= range; k++) {
					if (j - k >= 0) {
						for (int p = 0; p < vec_len; p++) {
							Xw[p] += dictionary[vocab_hash[GetIndexByWord(words_oneiter[j - k])]].wv[p];
						}
					}
					if (j + k < read_now) {
						for (int p = 0; p < vec_len; p++) {
							Xw[p] += dictionary[vocab_hash[GetIndexByWord(words_oneiter[j + k])]].wv[p];
						}
					}
				}
				int samples[7];//这里设置一个正样本和六个负样本
				samples[0] = vocab_hash[GetIndexByWord(words_oneiter[j])];//正样本
				for (int k = 1; k < 7; k++) {
					samples[k] = CreatNegSample(words_oneiter[j]);
				}

				for (int k = 0; k < 7; k++) {
					std::string w = words_oneiter[j];
					std::string u = dictionary[samples[k]].name;
					int L = (k == 0) ? 1 : 0;
					double Xwsita = 0;
					for (int p = 0; p < vec_len; p++) {
						Xwsita += Xw[p] * dictionary[samples[k]].sita[p];
					}
					double q = CalculateSigmoid(Xwsita);
					double g = learn_rate * (L - q);
					for (int p = 0; p < vec_len; p++) {
						e[p] += g * dictionary[samples[k]].sita[p];
					}
					for (int p = 0; p < vec_len; p++) {
						dictionary[samples[k]].sita[p] += g * Xw[p];
					}
				}

				//更新周边词语的词向量
				for (int k = 1; k <= range; k++) {
					if (j - k >= 0) {
						double* temp = dictionary[vocab_hash[GetIndexByWord(words_oneiter[j - k])]].wv;
						for (int p = 0; p < vec_len; p++) {
							temp[p] += e[p];
						}
					}
					if (j + k < read_now) {
						double* temp = dictionary[vocab_hash[GetIndexByWord(words_oneiter[j + k])]].wv;
						for (int p = 0; p < vec_len; p++) {
							temp[p] += e[p];
						}
					}
				}
			}
			read_now = 0;
		}
	}
}

int main() {
	//使用Hierarchical Softmax方法训练
	//生成Sigmoid值，并将其存储在数组sig_table中
	//InitSigTable();
	//std::cout << ">>初始化Sigmoid激活函数数组完毕" << std::endl;
	//memset(vocab_hash, -1, sizeof(int) * hash_size);//初始化hash数组
	//ReadOneLineFromFile_H();//从文件中读入文本语料，并进行神经网络的初始化工作
	//开始训练
	//trainCBOW_H();//CBOW
	//trainSKIPGRAM_H();//skip-gram


	//使用负采样训练
	InitSigTable();
	std::cout << ">>初始化Sigmoid激活函数数组完毕" << std::endl;
	memset(vocab_hash, -1, sizeof(int) * hash_size);//初始化hash数组
	ReadOneLineFromFile_N();//从文件中读入文本语料，并进行负采样数组的初始化工作
	trainCBOW_N();
	return 0;
}