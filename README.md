# yokoten 横展開ツール

カレントディレクトリ、または指定したフォルダをgrepし、対象となるファイルについて同一のスクリプトで生成AIに横展開修正の要否を質問します。

- Bedrockを経由したClaude v3.x
- Gemini

に対応しています。デフォルトではBedrockを使用します。

## モチベーション

2025/5/5現在のLLMは、トークン数の問題で複数のファイルを渡したりプロジェクト全体を俯瞰するタスクになると精度が下がる場合があります。  
一方で、単純な横展開タスクについては、特定の指示とファイルを1つ渡すことによって精度高く処理ができるはずです。  
そこで、grepで対象になったすべてのファイルについて、同じ指示でLLMから出力を受け取り、修正の要否や方針などを返却させるツールを作成しました。

## 準備

### インストール

```shell
git clone 
cd yokoten
pip3 install .
# pip install .
```

デフォルトと異なる設定で実行する場合は、yokoten/config.yaml.defaultの内容を参考にしてconfig.yamlを作成し、
~/.config/yokoten/config.yaml に配置してください。

### Bedrockを利用する場合

デフォルトでは `us-east-1` がリージョンとして設定されています。
異なるリージョンを使用する場合はconfig.yamlを作成してください。

Bedrockを使用できるIAMユーザーが必要です。
`aws configure`でIAMユーザーを設定してください。

### Geminiを利用する場合

環境変数GEMINI_API_KEYに、Geminiを利用可能なAPI KEYを設定してください。

## 実行コマンド

```shell
cd /path/to/your/repository
yokoten
```

または、直接main.pyを実行

```shell
cd yokoten/yokoten  # yokotenのリポジトリの中のyokotenフォルダに移動する
python3 main.py --input /path/to/your/repository
```

## その他

このツールの初期スクリプトはGemini 2.5 Proによって生成されました。
その後も、いくつかのLLMツールによって生成されたコードを利用しています。
