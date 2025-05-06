# yokoten 横展開ツール

カレントディレクトリ、または指定したフォルダをgit grepし、対象となるファイルについて同一のスクリプトでLLMに横展開修正の要否を質問します。
git grepで検索するため、対象はgit管理されている必要があります。git管理対象外のファイルは検索対象に含まれません。
以下のLLM(API)に対応しています。

- OpenAI
- Bedrockを経由したClaude v3.x
- Gemini

に対応しています。デフォルトではBedrockを使用します。

## モチベーション

2025/5/5現在のLLMは、トークン数の問題で複数のファイルを渡したりプロジェクト全体を俯瞰するタスクになると精度が下がる場合があります。  
一方で、単純な横展開タスクについては、特定の指示とファイルを1つ渡すことによって精度高く処理ができるはずです。  
そこで、git grepで対象になったすべてのファイルについて、同じ指示でLLMから出力を受け取り、修正の要否や方針などを返却させるツールを作成しました。

## 準備

### インストール

実行にはpython3.9以上（3.8でも動作します）とgitが必要です。

```shell
git clone https://github.com/sasanquaneuf/yokoten.git
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

なお、claude 3.7を使用するためには、model_idにInference profileのARNを指定する必要があります。
Inference profileは自分で作成する必要があるので注意ください。  
例）  
mode_id: arn:aws:bedrock:us-east-1:123456789012:inference-profile/us.anthropic.claude-3-7-sonnet-20250219-v1:0

### Geminiを利用する場合

環境変数`GEMINI_API_KEY`に、Google AI Studioで取得したAPI KEYを設定してください。  
（Google AI Studioページの「API キーを取得」(Get API key) や「APIキーを管理」(Manage API keys) のようなリンクから取得できます。）

### OpenAIを利用する場合

環境変数`OPENAI_API_KEY`に、API KEYを設定してください。  
（OpenAIのプラットフォームサイト https://platform.openai.com/ にログインした後にAPI keysページで取得できます。
なお、先に課金していないと429エラーになります。）

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

## 仕様

git管理化にないファイルは検索対象から除外されます。.gitignoreされているファイルや、untrackedなファイルなど。

## その他

このツールの初期スクリプトはGemini 2.5 Proによって生成されました。
その後も、いくつかのLLMツールによって生成されたコードを利用しています。
