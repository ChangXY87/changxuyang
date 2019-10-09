from pyltp import Segmentor, Postagger, Parser, NamedEntityRecognizer, SementicRoleLabeller
import re
import gensim
import numpy as np
from gensim.models import Word2Vec,word2vec
from scipy.linalg import norm

say_words = ['诊断', '交代', '说', '说道', '指出','报道','报道说','称', '警告',
           '所说', '告诉', '声称', '表示', '时说', '地说', '却说', '问道', '写道',
           '答道', '感叹', '谈到', '说出', '认为', '提到', '强调', '宣称', '表明',
           '明确指出', '所言', '所述', '所称', '所指', '常说', '断言', '名言', '告知',
           '询问', '知道', '得知', '质问', '问', '告诫', '坚称', '辩称', '否认', '还称',
           '指责', '透露', '坦言', '表达', '中说', '中称', '他称', '地问', '地称', '地用',
           '地指', '脱口而出', '一脸', '直说', '说好', '反问', '责怪', '放过', '慨叹', '问起',
           '喊道', '写到', '如是说', '何况', '答', '叹道', '岂能', '感慨', '叹', '赞叹', '叹息',
           '自叹', '自言', '谈及', '谈起', '谈论', '特别强调', '提及', '坦白', '相信', '看来',
           '觉得', '并不认为', '确信', '提过', '引用', '详细描述', '详述', '重申', '阐述', '阐释',
           '承认', '说明', '证实', '揭示', '自述', '直言', '深信', '断定', '获知', '知悉', '得悉',
           '透漏', '追问', '明白', '知晓', '发觉', '察觉到', '察觉', '怒斥', '斥责', '痛斥', '指摘',
           '回答', '请问', '坚信', '一再强调', '矢口否认', '反指', '坦承', '指证', '供称', '驳斥',
           '反驳', '指控', '澄清', '谴责', '批评', '抨击', '严厉批评', '诋毁', '责难', '忍不住',
           '大骂', '痛骂', '问及', '阐明']
model = Word2Vec.load('news_word2v-model.model')

class Ltp_parser:
    def __init__(self):
        self.segmentor = Segmentor()
        self.segmentor.load('/home/student/project/project-01/Four-Little-Frogs/ltp_data_v3.4.0/cws.model')
        self.postagger = Postagger()
        self.postagger.load('/home/student/project/project-01/Four-Little-Frogs/ltp_data_v3.4.0/pos.model')
        self.parser = Parser()
        self.parser.load('/home/student/project/project-01/Four-Little-Frogs/ltp_data_v3.4.0/parser.model')
        self.recognizer = NamedEntityRecognizer()
        self.recognizer.load('/home/student/project/project-01/Four-Little-Frogs/ltp_data_v3.4.0/ner.model')
        self.labeller = SementicRoleLabeller()
        self.labeller.load('/home/student/project/project-01/Four-Little-Frogs/ltp_data_v3.4.0/pisrl.model')

    '''依存句法分析'''
    def get_parser(self, words, postags):
        arcs = self.parser.parse(words, postags)
        # arcs = ' '.join("%d:%s" % (arc.head, arc.relation) for arc in arcs)
        return arcs

    '''命名实体识别'''
    def get_name_entity(self, words, postags):
        netags = self.recognizer.recognize(words, postags)
        netags = list(netags)
        return netags

    '''ltp模型释放'''
    def ltp_release(self):
        self.segmentor.release()
        self.postagger.release()
        self.parser.release()
        self.recognizer.release()

    '''LTP主函数'''
    def parser_main(self, sentence):
        words = list(self.segmentor.segment(sentence))
        postags = list(self.postagger.postag(words))
        arcs = self.get_parser(words, postags)
        netags = self.get_name_entity(words, postags)
        return words, postags, arcs, netags

class Extractor:
    def __init__(self):
        self.ltp = Ltp_parser()

    '''文章分句处理, 切分长句，冒号，分号，感叹号等做切分标识'''
    def split_sents(self, content):
        return [sentence for sentence in re.split(r'[？?！!。；\n\r]', content) if sentence]

    '''获取主语部分'''
    def get_name(self, name, predic, words, property, ne):
        index = words.index(name)
        cut_property = property[index+1:]
        pre = words[:index]
        pos = words[index+1:]
        while pre:
            w = pre.pop(-1)
            w_index = words.index(w)
            if property[w_index] == 'ADV': continue
            if property[w_index] in ['WP', 'ATT', 'SVB'] and (w not in ['，','。','、','）','（']):
                name = w + name
            else:
                pre = False
        while pos:
            w = pos.pop(0)
            p = cut_property.pop(0)
            if p in ['WP', 'LAD', 'COO', 'RAD'] and w != predic and (w not in ['，', '。', '、', '）', '（']):
                name = name + w
            else:
                return name
        return name

    '''获取言论信息'''
    def get_saying(self, sentence, proper, heads, pos):
        if '：' in sentence:
            return ''.join(sentence[sentence.index('：') + 1:])
        while pos < len(sentence):
            w = sentence[pos]
            p = proper[pos]
            h = heads[pos]
            # 谓语尚未结束
            if p in ['DBL', 'CMP', 'RAD']:
                pos += 1
                continue
            # 定语
            if p == 'ATT' and proper[h - 1] != 'SBV':
                pos = h
                continue
            # 宾语
            if p == 'VOB':
                pos += 1
                continue
            else:
                if w == '，':
                    return ''.join(sentence[pos + 1:])
                else:
                    return ''.join(sentence[pos:])

    '''基于Word2Vec的句子相似度匹配'''
    def cut_word(self, sentence):
        segmentor = Segmentor()
        segmentor.load('/home/student/project/project-01/Four-Little-Frogs/ltp_data_v3.4.0/cws.model')
        words = list(segmentor.segment(sentence))
        segmentor.release()
        # cut_words = ("\t".join(words))
        return words
    def vector_similarity(self, s1, s2):
        def sentence_vector(s):
            words = self.cut_word(s)
            v = np.zeros(100)
            for word in words:
                if word not in model:
                    word = '0'
                else:
                    v += model[word]
            v /= len(words)
            return v
        v1, v2 = sentence_vector(s1), sentence_vector(s2)
        return np.dot(v1, v2) / (norm(v1) * norm(v2))
    def get_new_sentenses(self, content):
        sentences = self.split_sents(content)
        expect = 0.7
        i = len(sentences)
        new_sentences = []

        while i > 1:
            s1 = sentences.pop(0)
            s2 = sentences[0]
            if self.vector_similarity(s1, s2) > 0.7:
                sentences[0] = s1 + ',' + s2
            else:
                new_sentences.append(s1)
            i -= 1
            if i == 1:
                if sentences[0] == s2:
                    new_sentences.append(s2)
                else:
                    new_sentences.append(sentences[0])
        return new_sentences

    '''获取信息主函数'''
    def main_parse_sentence(self, content):
        sentences = self.get_new_sentenses(content)
        names = []
        sayings = []
        for sentence in sentences:
            cut_words, postags, arcs, netags = self.ltp.parser_main(sentence)
            choose_word = [word for word in cut_words if word in say_words]
            if not choose_word:
                continue
            arcs_relation = [arc.relation for arc in arcs]
            stack = []
            for k, v in enumerate(arcs):
                if postags[k] in ['ni', 'nh']:
                    stack.append(cut_words[k])
                if v.relation == 'SBV' and (cut_words[v.head-1] in choose_word):
                    name = self.get_name(cut_words[k], arcs_relation[v.head-1], cut_words, arcs_relation, netags)
                    saying = self.get_saying(cut_words, arcs_relation, [i.head for i in arcs], v.head)
                    if not saying:
                        quotations = re.findall(r'“(.+?)”', sentence)
                        if quotations: says = quotations[-1]
                    names.append(name)
                    sayings.append(saying)
                    # return name, saying
                if cut_words[k] == '：':
                    name = stack.pop()
                    saying = ''.join(cut_words[k + 1:])
                    # return name, saying
                    names.append(name)
                    sayings.append(saying)

        self.ltp.ltp_release()
        return dict(zip(names, sayings))


#函数调用
content = '北京时间8日凌晨，记者获悉，标普道琼斯指数公司决定，1099只中国A股正式纳入标普新兴市场全球基准指数，该决定于9月23日开盘时生效。它们以25%的纳入因子纳入之后，预计A股在标普新兴市场全球基准指数中所占权重为6.2%，预计中国市场整体（含A股、港股、海外上市中概股）在该指数占权重36%。标普道琼斯指数表示，在9月23日生效日之前，纳入名单和权重还会有所变化，投资者仍以最终名单为准。此前，标普道琼斯指数公司指数投资策略全球主管、董事总经理克雷格·拉扎拉(Craig Lazzara)在接受媒体采访时表示，入选标普标准主要有两条：一是公司自由流通市值须达一亿美元，二是股票具备充分流动性。而流动性方面，他认为，在新兴市场当中，个股的流动性须达市值的10%，成熟市场当中该比例为20%。值得注意的是，此次，纳入名单中不含创业板股票，标普道琼斯指数官网表示，纳入创业板股票前，会就此再向市场参与者征询意见。作为全球最大的金融市场指数提供商之一，标普道琼斯指数纳入A股过程总体颇为顺利。根据此前安排，时间表如下：2018年12月5日，宣布中国A股将被入该指数。2018年12月31日，宣布A股入围初筛名单。2019年9月7日，初步入选个股名单变更至1099只。2019年9月23日将生效，届时标普道琼斯指数将正式纳入合格的相关标的。2018年12月5日，标普道琼斯指数公司对外宣布，将可以通过沪港通、深港通机制进行交易的合格中国A股纳入其有新兴市场分类的全球基准指数，纳入将自2019年9月23日市场开盘前生效，分类为“新兴市场”。当年底，标普道琼斯揭晓了A股入围的初筛名单，当时共有1241只A股股票入选。标普道琼斯指数公司此前表示，将会持续密切监测中国A股市场，并可能就进一步增加A股在全球基准中的权重。'
extractor = Extractor()
print(extractor.main_parse_sentence(content))


