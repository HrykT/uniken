初版 2016/9/6


【概要】
ユニシス研究会　「センサーを利用した行動分析」
センサーから採取したデータと行動内容の関連を機械学習させる。
なお現在は、手法の選択やデータ洗練方法の調査を主な目的としている。

【言語・ライブラリ】
Python2.7
＜主な使用ライブラリ＞
scikit-learn
numpy
pandas
matplotlib

【各プログラムの内容】

＜フォルダ構成と概要＞

ML:
├─datas                  …生データと加工後データを保存するフォルダ
│  InitDataProcess.py     …生データの加工用クラスを記述
│  Single_Models.py       …各学習モデルの基本的な処理をまとめたクラスを記述
│  Plot_predict.py        …データと予想結果をプロットして可視化する
│  MyPCA.py               …PCAによる特徴量抽出
│  sample_IniteDataProcess.py …IniteDataProcessの使用サンプル
│  sample_singlemodels.py     …Single_Modelsの使用サンプル