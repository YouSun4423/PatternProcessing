パターン情報処理 第二回レポート
 
# Requirement

* numpy = "1.23.3"
* opencv-python = "4.6.0.66"
* scipy = "*"
 
# Note

* images/Parrots.bmp : 標準画像データベース SIDBA からダウンロードした画像データ
* practice_i.py : 課題のi問目に回答したプログラム
* out ... 各プログラムのアウトプット先のディレクトリ
* out/practice_1/.. : SIDBA からダウンロードした画像データに平均化フィルタ、ガウシアンフィルタ、メジアンフィルタをかけた画像
* out/practice_2/out_blur.bmp,output.bmp : ガウシアンフィルタをかけて劣化させた画像と正則化逆フィルタで復元した画像
* out/practice_3/output.bmp : 疎行列の計算で256x256のサイズについて復元したもの
 
* 課題2については，MacBook Air (M1, 2020) 8Gにおいて，128x128のサイズまで復元可能でした．256x256はTimeoutしました．
* 課題3については，上記動作環境において，256x256のサイズの画像に対し190秒ほどで復元を終えました．

# Author

* 作成者 : 矢口悠月
* 学籍番号 : 21TE007
* 所属 : 埼玉大学工学部電気電子物理工学科2年生