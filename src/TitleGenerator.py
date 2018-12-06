

import numpy as np
import MeCab
from gensim.models import word2vec

# モデルのインポート
model = word2vec.Word2Vec.load("../latest-ja-word2vec-gensim-model/word2vec.gensim.model")

# 中二病コーパスリストを準備
cps = ["アイデンティティ","アルケー","アンチノミー","ウロボロス","ヴァルハラ","カタルシス","クライシス","サザンクロス","サンクチュアリ","シニカル","ジェネシス","ジェイド","タリスマン","デイズ","ディーヴァ","パルス","プラトニック","プリマドンナ","ミラージュ","メモワール","メロウ","ユグドラシル","リフレイン","リリカル","メシア","アビス","アポカリプス","ヴォイド","オーメン","サタン","シンドローム","スーパーノヴァ","ドッペルゲンガー","ナイトメア","ネクロマンサー","パラノイア","ファントム","ブラッディ","ヘルヘイム","マリオネット","メランコリー","ラビリンス","ルナティック","ルシフェル","レクイエム"]

# 形態素解析準備
t = MeCab.Tagger("-Ochasen")

# テキストのベクトルを計算
def get_vector(text):
    sum_vec = np.zeros(50)
    word_count = 0
    node = t.parseToNode(text)
    while node:
        fields = node.feature.split(",")
        # 名詞，動詞，形容詞に限定
        if fields[0] == '名詞' or fields[0] == '動詞' or fields[0] == '形容詞':
            sum_vec += model.wv[node.surface]
            word_count += 1
        node = node.next

    return sum_vec / word_count


# cos類似度を計算
def cos_sim(v1, v2):
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))


def TitleGenerator(text: str) -> str:
    # 入力文字列のベクトル化
    v_text = get_vector(text)
    
    # 中二病横文字から一番マッチする文字列を返す
    result_str = ""
    result_vec = 0

    for s in cps:
        cs = cos_sim(get_vector(s), v_text)
        if cs > result_vec:
            result_vec = cs
            result_str = s
    return "{}の{}".format(text, result_str)


if __name__ == '__main__':
    s = input()
    print(TitleGenerator(s))