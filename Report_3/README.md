パターン情報処理 第三回レポート
 
# Requirement

* opencv-python = "4.6.0.66"
* matplotlib = "3.6.0"
* sklearn = "*"
 
# Note

* images/Lenna.bmp : 標準画像データベース SIDBA からダウンロードした画像データ
* practice_i.py : 課題のi問目に回答したプログラム
* out ... 各プログラムのアウトプット先のディレクトリ
* out/practice_1/output.bmp :　SIDBからダウンロードした画像の一部をテンプレートとし，元の画像にガウシアンフィルタをかけてぼかした画像に対してテンプレートマッチングを行ったもの
* out/practice_2/output{i}.png : 手書き文字データセットMNISTから特定の数字（0、1など）の画像だけを抜き出して主成分分析を行った際の主成分の画像
* out/practice_3/output{i}.png  : 上位D個の主成分に対応するスコアから復元した元のデータの画像

# Author

* 作成者 : 矢口悠月
* 学籍番号 : 21TE007
* 所属 : 埼玉大学工学部電気電子物理工学科2年生