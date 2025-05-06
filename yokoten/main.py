import argparse
from datetime import datetime
import json
import os
from pathlib import Path
import subprocess
import sys
import time
import traceback
from typing import Dict, Any, Tuple, List, Optional, Union

import yaml


def load_config() -> Dict[str, Any]:
    """
    設定ファイル(config.yaml)を読み込む
    :return:
    """
    candidates = [
        Path.home() / ".config/yokoten/config.yaml",
        Path(__file__).parent / "config.yaml",
        Path(__file__).parent / "config.yaml.default",
    ]
    path: Optional[Path] = None
    try:
        config: dict = {}
        for path in candidates:
            if path.exists():
                with open(path, 'r', encoding='utf-8') as f:
                    config = yaml.safe_load(f)
                    break
        # 環境変数からAPIキーを読み込む (google_aiの場合)
        if config.get('ai_model', {}).get('platform') in ('google_ai', 'openai'):
            api_key_env = config.get('ai_model', {}).get('api_key_env')
            if api_key_env:
                api_key = os.environ.get(api_key_env)
                if not api_key:
                    print(f"エラー: 環境変数 '{api_key_env}' が設定されていません。", file=sys.stderr)
                    sys.exit(1)
                config['ai_model']['api_key'] = api_key
            else:
                print("エラー: config.yamlのai_modelセクションに'api_key_env'が設定されていません。", file=sys.stderr)
                sys.exit(1)

        # Vertex AI の場合の初期設定は initialize_ai で行う
        return config
    except FileNotFoundError:
        print(f"エラー: 設定ファイル '{candidates}' が見つかりません。", file=sys.stderr)
        traceback.print_exc()
        sys.exit(1)
    except yaml.YAMLError as e:
        print(f"エラー: 設定ファイル '{path}' の形式が正しくありません: {e}", file=sys.stderr)
        traceback.print_exc()
        sys.exit(1)
    except Exception as e:
        print(f"エラー: 設定ファイルの読み込み中に予期せぬエラーが発生しました: {e}", file=sys.stderr)
        traceback.print_exc()
        sys.exit(1)


def initialize_ai(config: Dict[str, Any]) -> Tuple[Dict[str, Any], str]:
    """
    設定に基づいてAIクライアントを初期化する
    :param config: dict 設定データ
    :return:
    """
    ai_config = config.get('ai_model', {})
    platform = ai_config.get('platform')

    if platform == "google_ai":
        try:
            import google.generativeai as genai
            print("Google AI (Gemini) SDK を使用します。")
            genai.configure(api_key=ai_config['api_key'])
            model = genai.GenerativeModel(ai_config['model_name'])
            return {'model': model}, "google_ai"
        except ImportError:
            print("エラー: 'google-generativeai' ライブラリがインストールされていません。", file=sys.stderr)
            print("pip install google-generativeai を実行してください。", file=sys.stderr)
            traceback.print_exc()
            sys.exit(1)
        except Exception as e:
            print(f"エラー: Google AI クライアントの初期化に失敗しました: {e}", file=sys.stderr)
            traceback.print_exc()
            sys.exit(1)

    elif platform == "vertex_ai":
        try:
            import vertexai
            from vertexai.generative_models import GenerativeModel
            print("Google Cloud Vertex AI SDK を使用します。")
            project_id = ai_config.get('project_id')
            location = ai_config.get('location')
            if not project_id or not location:
                print("エラー: Vertex AI を使用するには config.yaml で project_id と location を設定してください。", file=sys.stderr)
                sys.exit(1)
            vertexai.init(project=project_id, location=location)
            model = GenerativeModel(ai_config['model_name'])
            return {'model': model}, "vertex_ai"
        except ImportError:
            print("エラー: 'google-cloud-aiplatform' ライブラリがインストールされていません。", file=sys.stderr)
            print("pip install google-cloud-aiplatform を実行してください。", file=sys.stderr)
            traceback.print_exc()
            sys.exit(1)
        except Exception as e:
            print(f"エラー: Vertex AI クライアントの初期化に失敗しました: {e}", file=sys.stderr)
            traceback.print_exc()
            sys.exit(1)

    elif platform == "amazon_bedrock":
        try:
            import boto3
            from botocore.exceptions import BotoCoreError

            model_id = ai_config.get('model_id')
            region = ai_config.get('region', 'us-east-1')  # デフォルトリージョン
            if not model_id:
                print("エラー: Amazon Bedrock を使用するには 'model_id' を config.yaml に設定してください。", file=sys.stderr)
                sys.exit(1)

            bedrock_runtime = boto3.client("bedrock-runtime", region_name=region)
            print("Amazon Bedrock SDK を使用します。")
            return {"client": bedrock_runtime, "model_id": model_id}, "amazon_bedrock"
        except ImportError:
            print("エラー: 'boto3' ライブラリがインストールされていません。", file=sys.stderr)
            print("pip install boto3 を実行してください。", file=sys.stderr)
            traceback.print_exc()
            sys.exit(1)
        except Exception as e:
            print(f"エラー: Bedrock クライアントの初期化に失敗しました: {e}", file=sys.stderr)
            traceback.print_exc()
            sys.exit(1)

    elif platform == 'openai':
        try:
            from openai import OpenAI
            from openai import APIError, RateLimitError, APIConnectionError, APITimeoutError
            print("OpenAI SDK を使用します。")
            client = OpenAI(api_key=ai_config['api_key'])
            # model_name は query_ai_for_file で使用する
            return {"client": client, "model_name": ai_config['model_name']}, "openai"
        except ImportError:
            print("エラー: 'openai' ライブラリがインストールされていません。", file=sys.stderr)
            print("pip install openai を実行してください。", file=sys.stderr)
            traceback.print_exc()
            sys.exit(1)
        except Exception as e:
            print(f"エラー: OpenAI クライアントの初期化に失敗しました: {e}", file=sys.stderr)
            traceback.print_exc()
            sys.exit(1)

    else:
        print(f"エラー: 未知のプラットフォーム '{platform}' が設定されています。", file=sys.stderr)
        sys.exit(1)


def run_grep(input_path: str, pattern: str) -> List[str]:
    """
    Git リポジトリ配下で git grep を実行し、ヒットしたファイルのパス一覧を返す

    :param input_path: str 検索対象ディレクトリ（Git リポジトリのルートまたはそのサブディレクトリ）
    :param pattern: str 検索する文字列
    :return: List[str] ヒットしたファイルパス
    """
    if not os.path.exists(input_path):
        print(f"警告: パス '{input_path}' が存在しません。スキップします。", file=sys.stderr)
        return []

    # まず Git リポジトリかどうか確認
    try:
        result_check = subprocess.run(
            ['git', "-C", input_path, "rev-parse", "--is-inside-work-tree"],
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="ignore",
            check=False,
        )
        if result_check.returncode != 0 or result_check.stdout.strip() != "true":
            print(f"警告: '{input_path}' は Git リポジトリではありません。", file=sys.stderr)
            return []
    except FileNotFoundError:
        print(
            f"エラー: git コマンドが見つかりません。"
            f"パスを確認するか、config.yaml の git_command を修正してください。",
            file=sys.stderr,
        )
        return []
    except Exception as e:
        print(f"エラー: git rev-parse 実行中に予期せぬエラーが発生しました: {e}", file=sys.stderr)
        return []

    # git grep 実行 (-I バイナリ除外, -l ファイル名のみ, --null 出力を NUL 区切り)
    try:
        result = subprocess.run(
            [
                'git',
                "-C",
                input_path,
                "grep",
                "-Il",          # -I: バイナリ省く, -l: ファイル名のみ
                "--null",       # 区切り文字を \0 にして安全に分割
                pattern,
                "--",           # ここ以降はパス指定
                "."             # input_path 配下すべて
            ],
            capture_output=True,
            text=False,        # バイナリセーフに扱うため bytes で受け取る
            check=False,
        )

        if result.returncode > 1:  # 0: 見つかった, 1: 見つからず, >1: エラー
            stderr_text = result.stderr.decode("utf-8", errors="ignore") if result.stderr else ""
            print(
                f"警告: git grep 実行中にエラーが発生しました "
                f"(パス: {input_path}, パターン: {pattern}).\nstderr:\n{stderr_text}",
                file=sys.stderr,
            )

        # NUL 区切りなので split(b'\0')
        files = [
            os.path.join(input_path, f.decode("utf-8", errors="ignore"))
            for f in result.stdout.split(b"\0")
            if f
        ]
        valid_files = [f for f in files if os.path.isfile(f)]
        print(f"パス '{input_path}' で {len(valid_files)} 件のファイルがヒットしました。")
        return valid_files

    except FileNotFoundError:
        print(
            f"エラー: git コマンドが見つかりません。"
            f"パスを確認するか、config.yaml の git_command を修正してください。",
            file=sys.stderr,
        )
        return []
    except Exception as e:
        print(f"エラー: git grep 実行中に予期せぬエラーが発生しました (パス: {input_path}): {e}", file=sys.stderr)
        return []


def read_file_content(filepath: str) -> Optional[str]:
    """
    ファイルの内容を読み込む
    :param filepath:
    :return:
    """
    try:
        with open(filepath, 'r', encoding='utf-8', errors='ignore') as f:
            return f.read()
    except Exception as e:
        print(f"警告: ファイル '{filepath}' の読み込みに失敗しました: {e}", file=sys.stderr)
        return None  # エラーの場合はNoneを返す


def query_ai_for_file(
        model: Dict[str, Any], platform: str, file_path: str, file_content: str,
        default_prompt: str, additional_prompt: str, max_retries: int) -> Dict[str, Union[bool, str, None]]:
    """
    AIにファイルを渡し、JSON形式の回答を得る。リトライ機能付き。
    :param model:
    :param platform:
    :param file_path:
    :param file_content:
    :param default_prompt:
    :param additional_prompt:
    :param max_retries:
    :return:
    """
    user_prompt = f"{additional_prompt}\n\n" \
                  f"--- 対象ファイルの内容 ({os.path.basename(file_path)}) ---\n" \
                  f"```\n{file_content}\n```\n--- ファイル内容ここまで ---\n\n分析結果をJSONで出力してください。"

    for attempt in range(max_retries):
        response_text = ''
        try:
            print(f"  Attempt {attempt + 1}/{max_retries}...")
            if platform == "google_ai":
                response = model['model'].generate_content(f"{default_prompt}\n\n{user_prompt}")
                response_text = response.text
            elif platform == "vertex_ai":
                response = model['model'].generate_content(f"{default_prompt}\n\n{user_prompt}")
                response_text = response.text  # Vertex AIも .text でアクセス可能

            elif platform == "amazon_bedrock":
                try:
                    messages = [
                        {
                            "role": "user",
                            "content": [{"text": f"{default_prompt}\n\n{user_prompt}"}]
                        }
                    ]

                    response = model["client"].converse(
                        modelId=model['model_id'],
                        messages=messages
                    )
                    # レスポンスからアシスタントの応答を取得
                    response_text = response['output']['message']['content'][0]['text'].replace('\n', ' ')

                except Exception as e:
                    print(f"  エラー: Bedrock からの応答取得に失敗しました: {e}")
                    response_text = ''

            elif platform == "openai":
                # model = {"client": OpenAIクライアント, "model_name": str}
                client = model["client"]
                model_name = model["model_name"]
                # OpenAI API呼び出し
                response = client.chat.completions.create(
                    model=model_name,
                    messages=[
                        {"role": "system", "content": default_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    # JSONモードを試みる (モデルが対応している場合)
                    # response_format={"type": "json_object"}
                    # 注意: JSONモードが失敗する場合や使えないモデルもあるため、
                    # まずはテキストで取得し、後でパースする方が堅牢
                    temperature=0.2,  # 再現性を高めるために低めに設定
                )
                response_text = response.choices[0].message.content
            else:
                raise ValueError(f"未知のプラットフォーム: {platform}")

            # 回答が ```json ... ``` で囲まれている場合に対応
            if response_text.strip().startswith("```json"):
                response_text = response_text.strip()[7:-3].strip()
            elif response_text.strip().startswith("```"):
                response_text = response_text.strip()[3:-3].strip()

            # JSONパース試行
            result_json = json.loads(response_text)

            # 必要なキーと型を検証
            if isinstance(result_json, dict) and \
               'required' in result_json and isinstance(result_json['required'], bool) and \
               'details' in result_json and isinstance(result_json['details'], str):
                print("  分析成功 (JSON形式OK)")
                return result_json  # 成功したらJSONを返す
            else:
                print(f"  警告: AIの応答が期待されるJSON形式ではありません。応答: {response_text[:200]}...")  # 長すぎる場合は切り詰める

        except json.JSONDecodeError:
            print(f"  警告: AIの応答がJSONとしてパースできませんでした。応答: {response_text[:200]}...")
        except Exception as e:
            # APIエラーなどもここでキャッチ
            print(f"  エラー: AIへの問い合わせ中にエラーが発生しました: {e}")

        # リトライ前に少し待機
        if attempt < max_retries - 1:
            wait_time = 2 ** attempt  # Exponential backoff
            print(f"  {wait_time}秒待機してリトライします...")
            time.sleep(wait_time)

    print(f"  エラー: {max_retries}回試行しましたが、有効なJSON応答を得られませんでした。")
    return {"required": None, "details": "AIからの有効な応答を得られませんでした。"}  # エラーを示す情報を返す


def main():
    """
    メイン処理
    :return:
    """
    # 1. 設定読み込み
    config: Dict[str, Any] = load_config()
    default_prompt = config.get('prompts', {}).get('default_prefix', '')
    file_type = config.get('file_type', 'json')
    output_file = datetime.now().strftime(config.get('output_file', 'result_yokoten%Y%m%d%H%M%S')) + f'.{file_type}'
    max_retries = config.get('max_retries', 5)
    report_path_format = config.get('report_path_format', 'relative')

    # コマンドライン引数
    parser = argparse.ArgumentParser(description="横展開対象をgit grepで検索して、1ファイルずつLLMで分析するツール")
    parser.add_argument(
        '--input',
        nargs='*',
        help='検索対象のパス（指定しない場合はカレントディレクトリ）'
    )
    args = parser.parse_args()
    if args.input:
        input_paths = args.input
        report_path_format = 'absolute'
    else:
        input_paths = [str(Path.cwd())]

    if not default_prompt:
        print("警告: config.yamlにデフォルトプロンプト(default_prompt)が設定されていません。", file=sys.stderr)
        # 実行は継続するが、AIへの指示が不十分になる可能性

    # 2. AIクライアント初期化
    ai_model: Dict[str, Any]
    ai_model, platform = initialize_ai(config)

    # 3. grepパターンと追加プロンプトの入力
    print("-" * 30)
    grep_pattern = input("git grepするパターンを入力してください: ")
    if not grep_pattern:
        print("エラー: grepパターンが入力されていません。", file=sys.stderr)
        sys.exit(1)

    print("-" * 30)
    print("今回の横展開タスクに関する追加の指示を入力してください。")
    print("例: CVE-2024-XXXX の修正が他の箇所にも必要か確認してください。特にXXX関数呼び出し箇所を注視してください。")
    additional_prompt = input("> ")

    # 4. grep実行とファイルリスト作成
    print("-" * 30)
    print("grepしています...")
    all_found_files = []
    for path in input_paths:
        found_files = run_grep(path, grep_pattern)
        all_found_files.extend(found_files)

    if not all_found_files:
        print("指定されたパターンに一致するファイルは見つかりませんでした。")
        sys.exit(0)

    print("-" * 30)
    print(f"合計 {len(all_found_files)} 件のファイルが見つかりました。これらのファイルを分析します。")
    print("-" * 30)

    # 5. 各ファイルの分析と結果保存
    results: Dict[str, Dict[str, Union[bool, str, None]]] = {}
    required_files_list = []
    start_time = datetime.now()

    for i, file_path in enumerate(all_found_files):
        file_path_for_output = file_path
        if report_path_format == 'relative':
            for input_path in input_paths:
                if input_path == file_path_for_output[:len(input_path)]:
                    file_path_for_output = file_path_for_output[len(input_path) + 1:]
        print(f"[{i + 1}/{len(all_found_files)}] ファイル '{file_path_for_output}' を処理中...")

        # ファイル内容読み込み
        file_content = read_file_content(file_path)
        if file_content is None:
            results[file_path_for_output] = {"required": None, "details": "ファイル読み込みエラー"}
            print("  スキップ (読み込みエラー)")
            continue
        if not file_content.strip():
            results[file_path_for_output] = {"required": False, "details": "ファイルが空です。"}
            print("  スキップ (ファイルが空)")
            continue
        # AIに問い合わせ
        ai_result = query_ai_for_file(
            ai_model, platform, file_path, file_content, default_prompt, additional_prompt, max_retries
        )
        results[file_path_for_output] = ai_result

        # 対応が必要なファイルをリストアップ
        if ai_result.get("required") is True:  # 明示的にTrueの場合のみ
            required_files_list.append(file_path_for_output)
            print(f"  -> ★対応が必要と判断されました。")
        elif ai_result.get("required") is False:
            print(f"  -> 対応は不要と判断されました。")
        else:
            print(f"  -> 判断できませんでした（エラーまたは不明）。")

    end_time = datetime.now()
    print("-" * 30)
    print(f"分析完了 (処理時間: {end_time - start_time})")

    # 6. 結果の保存と表示
    try:
        ai_model = {}
        for key in config['ai_model'].keys():
            if key in ('platform', 'model_id', 'model_name'):
                ai_model[key] = config['ai_model'][key]
        prompts = {
            'default_prefix': config['prompts']['default_prefix'],
            'additional_prompt': additional_prompt,
        }
        save_data = {
            'start_time': start_time.strftime('%Y-%m-%d %H:%M:%S'),
            'ai_model': ai_model,
            'grep_pattern': grep_pattern,
            'prompts': prompts,
            'results': results
        }
        with open(output_file, 'w', encoding='utf-8') as f:
            if file_type == 'json':
                json.dump(save_data, f, indent=2, ensure_ascii=False)
            elif file_type == 'html':
                with open(Path(__file__).parent / "template/report.html", 'r') as template:
                    html = ''.join(template.readlines())
                    f.write(html.replace(
                        '{}/* json_data */',
                        json.dumps(save_data, indent=2, ensure_ascii=False).replace('<', '\\u003C')))
        print(f"詳細な分析結果を '{output_file}' に保存しました。")
    except Exception as e:
        print(f"エラー: 結果ファイル '{output_file}' の保存に失敗しました: {e}", file=sys.stderr)

    print("-" * 30)
    print("【横展開対応が必要な可能性のあるファイル一覧】")
    if required_files_list:
        for file_path in required_files_list:
            print(f"- {file_path}")
    else:
        print("対応が必要と判断されたファイルはありませんでした。")
        # 結果ファイルにはエラーやNoneで判断できなかったファイルも含まれるため、そちらも確認推奨

    print("-" * 30)
    print(f"簡易的な結果閲覧: '{output_file}' を確認してください。")
    print("スクリプトを終了します。")


if __name__ == "__main__":
    main()
