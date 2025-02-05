#!/usr/bin/python3
# -*- coding: utf-8 -*-
"""
Gocro3: Genuine OCR Optimizer 3
Author: Akihiko Fujita
Version: 0.1

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import fitz
import pytesseract
from PIL import Image, ImageEnhance, ImageFilter
import io
from reportlab.pdfgen import canvas
from reportlab.lib.pagesizes import letter
import tkinter as tk
from tkinter import filedialog, messagebox, simpledialog
import re
from typing import List, Tuple, Optional, Dict
from dataclasses import dataclass
import numpy as np
from concurrent.futures import ThreadPoolExecutor
import json
import logging
from datetime import datetime
import cv2

# Tesseractのパス設定
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# 強化されたログ設定
logging.basicConfig(
    filename=f'ocr_log_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log',
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

@dataclass
class OCRConfig:
    """追加のパラメータを含むOCR処理設定クラス"""
    lang: str
    psm: int
    oem: Optional[int] = None
    dpi: Optional[int] = 300
    whitelist: Optional[str] = None
    
    def get_config_string(self) -> str:
        """強化されたオプションでTesseractの設定文字列を生成"""
        config = f'-l {self.lang} --psm {self.psm}'
        if self.oem is not None:
            config += f' --oem {self.oem}'
        if self.whitelist:
            config += f' -c tessedit_char_whitelist={self.whitelist}'
        return config

class ImagePreprocessor:
    """追加のメソッドを含む強化された画像前処理クラス"""
    
    @staticmethod
    def apply_all_preprocessing(image: Image.Image) -> List[Tuple[str, Image.Image]]:
        """ラベル付きの複数の前処理画像バージョンを生成"""
        preprocessed_images = []
        
        # 基本的な前処理
        basic = ImagePreprocessor._basic_preprocessing(image)
        preprocessed_images.append(("basic", basic))
        
        # 高度な前処理
        advanced = ImagePreprocessor._advanced_preprocessing(image)
        preprocessed_images.append(("advanced", advanced))
        
        # [new] ノイズ除去前処理
        denoised = ImagePreprocessor._denoise_preprocessing(image)
        preprocessed_images.append(("denoised", denoised))
        
        # [new] 傾き補正前処理
        deskewed = ImagePreprocessor._deskew_preprocessing(image)
        preprocessed_images.append(("deskewed", deskewed))
        
        return preprocessed_images
    
    @staticmethod
    def _basic_preprocessing(image: Image.Image) -> Image.Image:
        """拡張されたリサイジングを伴う基本的な前処理パイプライン"""
        img = image.copy()
        img = img.resize((img.width * 2, img.height * 2), Image.LANCZOS)
        img = img.convert('L')
        img = img.point(lambda x: 0 if x < 128 else 255, '1')
        return img
    
    @staticmethod
    def _advanced_preprocessing(image: Image.Image) -> Image.Image:
        """強化されたコントラストとシャープ化を伴う高度な前処理"""
        img = image.copy()
        img = img.resize((img.width * 2, img.height * 2), Image.LANCZOS)
        img = img.convert('L')
        
        # 強化されたコントラスト
        enhancer = ImageEnhance.Contrast(img)
        img = enhancer.enhance(2.5)
        
        # 複数回のシャープ化
        for _ in range(3):
            img = img.filter(ImageFilter.SHARPEN)
        
        # 高度な適応的二値化
        img = img.point(lambda x: 0 if x < 140 else 255, '1')
        return img
    
    @staticmethod
    def _denoise_preprocessing(image: Image.Image) -> Image.Image:
        """[new] 高度なノイズ除去前処理"""
        img = image.copy()
        img = img.resize((img.width * 2, img.height * 2), Image.LANCZOS)
        
        # 高度な処理のためにOpenCV形式に変換
        cv_img = np.array(img)
        
        # 高度なノイズ除去を適用
        if len(cv_img.shape) == 3:
            cv_img = cv2.cvtColor(cv_img, cv2.COLOR_RGB2GRAY)
        cv_img = cv2.fastNlMeansDenoising(cv_img)
        
        # PILに戻す
        img = Image.fromarray(cv_img)
        
        # 最終的なコントラスト調整
        enhancer = ImageEnhance.Contrast(img)
        img = enhancer.enhance(1.8)
        
        return img
    
    @staticmethod
    def _deskew_preprocessing(image: Image.Image) -> Image.Image:
        """[new] 傾き補正前処理"""
        img = image.copy()
        
        # OpenCV形式に変換
        cv_img = np.array(img)
        if len(cv_img.shape) == 3:
            gray = cv2.cvtColor(cv_img, cv2.COLOR_RGB2GRAY)
        else:
            gray = cv_img
            
        # エッジの検出
        edges = cv2.Canny(gray, 50, 150, apertureSize=3)
        
        # ハフ変換を使用した線の検出
        lines = cv2.HoughLines(edges, 1, np.pi / 180, 100)
        
        if lines is not None:
            # 傾き角度の計算
            angles = []
            for rho, theta in lines[:, 0]:
                angle = np.degrees(theta)
                if angle < 45 or angle > 135:
                    angles.append(angle)
            
            if angles:
                # 中央角度を取得
                median_angle = np.median(angles) - 90
                
                # 画像を回転
                (h, w) = cv_img.shape[:2]
                center = (w // 2, h // 2)
                M = cv2.getRotationMatrix2D(center, median_angle, 1.0)
                cv_img = cv2.warpAffine(cv_img, M, (w, h),
                                      flags=cv2.INTER_CUBIC,
                                      borderMode=cv2.BORDER_REPLICATE)
        
        # PILに戻す
        return Image.fromarray(cv_img)

class OCREvaluator:
    """強化されたOCR評価クラス、および信頼度スコアの追加"""
    
    def __init__(self):
        self.configs = [
            OCRConfig(lang='jpn+eng', psm=3),
            OCRConfig(lang='jpn+eng', psm=6),
            OCRConfig(lang='jpn+eng', psm=11),
            OCRConfig(lang='jpn+eng', psm=4, oem=1),
            OCRConfig(lang='jpn+eng', psm=13),
            # [new] 特定の用途に対する追加設定
            OCRConfig(lang='jpn+eng', psm=6, whitelist='0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz'),
            OCRConfig(lang='jpn_vert+eng', psm=5)  # 縦書き日本語テキスト
        ]
        
        # 信頼度しきい値を初期化
        self.confidence_threshold = 60.0
    
    def process_image(self, image: Image.Image, use_japanese_word_count: bool) -> Tuple[str, float]:
        """複数の設定で画像を処理し、信頼度付きの最良結果を返す"""
        all_results = []
        
        # 各前処理画像バージョンを各設定で処理
        preprocessed_images = ImagePreprocessor.apply_all_preprocessing(image)
        
        with ThreadPoolExecutor() as executor:
            futures = []
            for label, img in preprocessed_images:
                for config in self.configs:
                    futures.append(
                        executor.submit(
                            self._process_single_config,
                            img,
                            config,
                            label
                        )
                    )
            
            for future in futures:
                result = future.result()
                if result:
                    all_results.append(result)
        
        # 結果をログに記録
        self._log_results(all_results)
        
        # 最良の結果を選択
        texts = [(result[2], result[3]) for result in all_results]  # (テキスト, 信頼度)
        best_text, confidence = self._select_best_result(texts, use_japanese_word_count)
        
        return best_text, confidence
    
    def _process_single_config(self, img: Image.Image, config: OCRConfig, preprocess_label: str) -> Optional[Tuple]:
        """単一の設定で画像を処理"""
        try:
            config_str = config.get_config_string()
            
            # 詳細なOCRデータを取得し、信頼度スコアを含む
            ocr_data = pytesseract.image_to_data(img, config=config_str, output_type=pytesseract.Output.DICT)
            
            # 平均信頼度を計算
            confidences = [float(conf) for conf in ocr_data['conf'] if conf != '-1']
            avg_confidence = np.mean(confidences) if confidences else 0.0
            
            # テキストを取得
            text = pytesseract.image_to_string(img, config=config_str)
            
            return (config_str, preprocess_label, text, avg_confidence)
        except Exception as e:
            logging.error(f"設定 {config_str} の処理でエラーが発生しました: {str(e)}")
            return None
    
    def _log_results(self, all_results: List[Tuple]):
        """詳細な情報を含むOCR結果をログに記録"""
        log_data = {
            'timestamp': datetime.now().isoformat(),
            'results': [
                {
                    'config': result[0],
                    'preprocess': result[1],
                    'confidence': result[3],
                    'text_sample': result[2][:100]  # 最初の100文字
                }
                for result in all_results
            ]
        }
        
        logging.info(f"OCR結果: {json.dumps(log_data, ensure_ascii=False, indent=2)}")
    
    def _select_best_result(self, results: List[Tuple[str, float]], use_japanese_word_count: bool) -> Tuple[str, float]:
        """信頼度とテキスト品質に基づいて最良の結果を選択"""
        if not results:
            return "", 0.0
        
        scored_results = []
        for text, confidence in results:
            if use_japanese_word_count:
                quality_score = self._evaluate_japanese_text_quality(text)
            else:
                quality_score = self._evaluate_general_text_quality(text)
            
            # 信頼度と品質スコアを組み合わせる
            final_score = 0.7 * confidence + 0.3 * quality_score * 100
            scored_results.append((text, final_score))
        
        # 最良の結果とそのスコアを返す
        best_result = max(scored_results, key=lambda x: x[1])
        return best_result

    @staticmethod
    def _evaluate_japanese_text_quality(text: str) -> float:
        """強化された日本語テキスト品質評価"""
        # 一般的な日本語の句読点と記号のチェック
        jp_chars = re.findall(r'[\u4e00-\u9fff\u3040-\u309f\u30a0-\u30ff\u3000-\u303f]', text)
        jp_char_count = len(jp_chars)
        total_char_count = len(text.strip())
        
        if total_char_count == 0:
            return 0
        
        # 日本語文字の割合を計算
        jp_ratio = jp_char_count / total_char_count
        
        # 漸進的なスケーリングを伴う長さ係数
        length_factor = 1.0 - (1.0 / (1.0 + total_char_count / 100))
        
        # 一貫性のある文字の組み合わせのチェック
        coherence_score = 0.0
        if jp_char_count > 1:
            valid_pairs = 0
            for i in range(len(jp_chars) - 1):
                # 日本語の文字対が有効かどうかをチェック
                if re.match(r'[\u3040-\u309f][\u3040-\u309f]|[\u30a0-\u30ff][\u30a0-\u30ff]|[\u4e00-\u9fff]', 
                          jp_chars[i] + jp_chars[i + 1]):
                    valid_pairs += 1
            coherence_score = valid_pairs / (len(jp_chars) - 1) if len(jp_chars) > 1 else 0
        
        # 重みを付けてスコアを組み合わせる
        final_score = (0.4 * jp_ratio + 0.3 * length_factor + 0.3 * coherence_score)
        
        return final_score
    
    @staticmethod
    def _evaluate_general_text_quality(text: str) -> float:
        """強化された一般的なテキスト品質評価"""
        # テキストを正規化
        cleaned_text = ' '.join(text.split())
        words = cleaned_text.split()
        
        if not words:
            return 0
        
        # 基本的な指標を計算
        word_count = len(words)
        char_count = len(cleaned_text)
        avg_word_length = char_count / word_count
        
        # 妥当な単語の長さを持つかチェック
        reasonable_words = sum(1 for word in words if 2 <= len(word) <= 15)
        word_length_score = reasonable_words / word_count
        
        # 一般的な単語パターンのチェック
        pattern_score = 0
        for i in range(len(words) - 1):
            # 一般的な単語の組み合わせのチェック (冠詞、前置詞など)
            if re.match(r'^(the|a|an|in|on|at|to|for|of|with)\s+\w+', 
                       words[i].lower() + ' ' + words[i + 1].lower()):
                pattern_score += 1
        pattern_score = pattern_score / (len(words) - 1) if len(words) > 1 else 0
        
        # 重みを付けて最終的なスコアを算出
        final_score = (0.4 * word_length_score + 
                      0.3 * min(1.0, char_count / 500) +  # 長さ係数
                      0.3 * pattern_score)
        
        return final_score

class PDFProcessor:
    """PDF処理"""
    
    def __init__(self):
        self.ocr_evaluator = OCREvaluator()
        self.max_threads = 4
    
    def process_pdf(self, input_path: str, output_path: str, use_japanese_word_count: bool):
        """高度な機能と並列処理を用いてPDFを処理"""
        pdf_document = fitz.open(input_path)
        num_pages = pdf_document.page_count
        
        # メタデータを含む出力PDFを準備
        packet = io.BytesIO()
        can = canvas.Canvas(packet, pagesize=letter)
        
        # ページを並列に処理
        with ThreadPoolExecutor(max_workers=self.max_threads) as executor:
            future_to_page = {
                executor.submit(
                    self._process_single_page,
                    pdf_document,
                    page_num,
                    use_japanese_word_count
                ): page_num
                for page_num in range(num_pages)
            }
            
            # 順序を維持しながら結果を収集
            results = [None] * num_pages
            for future in futures.as_completed(future_to_page):
                page_num = future_to_page[future]
                try:
                    text, confidence = future.result()
                    results[page_num] = (text, confidence)
                    logging.info(f"ページ {page_num + 1}/{num_pages} が信頼度: {confidence:.2f} で処理されました")
                except Exception as e:
                    logging.error(f"ページ {page_num + 1} の処理でエラー: {str(e)}")
                    results[page_num] = ("ページ処理エラー", 0.0)
        
        # 処理されたテキストを出力PDFに追加
        for page_num, (text, confidence) in enumerate(results):
            # フォーマットを含むテキストの配置を強化
            self._add_text_to_pdf(can, text, page_num, confidence)
            
        can.save()
        packet.seek(0)
        
        # 強化されたメタデータを含むPDFの結合
        self._combine_pdfs(pdf_document, packet, output_path, results)

def _process_single_page(self, pdf_document, page_num: int, use_japanese_word_count: bool) -> Tuple[str, float]:
        """画像抽出を用いて単一のPDFページを処理"""
        page = pdf_document[page_num]
        
        # 最適な設定で高品質のピックスマップを取得
        pix = page.get_pixmap(matrix=fitz.Matrix(2.0, 2.0), alpha=False)
        img = Image.open(io.BytesIO(pix.tobytes()))
        
        # 画像を処理し、信頼度スコア付きのテキストを取得
        return self.ocr_evaluator.process_image(img, use_japanese_word_count)

    def _add_text_to_pdf(self, canvas_obj, text: str, page_num: int, confidence: float):
        """フォーマットを強化してPDFにテキストを追加"""
        # 最適なテキスト配置を計算
        y_position = 750 - page_num * 50
        
        # しきい値以下の信頼度の場合、信頼度インジケータを追加
        if confidence < 0.7:
            canvas_obj.setFillColorRGB(0.7, 0, 0)  # 低信頼度を表す赤色
            canvas_obj.drawString(10, y_position + 12, f"低信頼度: {confidence:.2f}")
            canvas_obj.setFillColorRGB(0, 0, 0)  # 黒にリセット
        
        # 適切なラッピング付きで処理済みテキストを追加
        wrapped_text = textwrap.fill(text[:1000], width=100)
        for i, line in enumerate(wrapped_text.split('\n')):
            canvas_obj.drawString(10, y_position - (i * 12), line)

    def _combine_pdfs(self, original_pdf, ocr_pdf, output_path: str, results: List[Tuple[str, float]]):
        """強化されたメタデータおよびブックマーク付きでPDFを結合"""
        new_pdf = fitz.open("pdf", ocr_pdf.read())
        output = fitz.open()
        
        # メタデータを追加
        output.set_metadata({
            'title': 'OCR Enhanced Document',
            'creator': 'Gocro OCR Optimizer',
            'producer': 'Enhanced PDF Processor',
            'creationDate': datetime.now().strftime('%Y%m%d%H%M%S')
        })
        
        # ブックマーク付きでページを結合
        for page_num in range(len(results)):
            # 元のページを追加
            output.insert_pdf(original_pdf, from_page=page_num, to_page=page_num)
            
            # OCRレイヤーを追加
            output.insert_pdf(new_pdf, from_page=page_num, to_page=page_num)
            
            # 各ページに信頼度情報付きのブックマークを追加
            confidence = results[page_num][1]
            output.set_toc([
                [1, f"ページ {page_num + 1} (信頼度: {confidence:.2f})", page_num + 1]
            ])
        
        # 最適化して保存
        output.save(output_path, garbage=3, deflate=True)

# ユーザーインターフェイスクラスを強化
class UserInterface:
    """追加オプションを備えた強化されたユーザーインターフェイス"""
    
    @staticmethod
    def get_input_file() -> Optional[str]:
        """拡張ファイルダイアログで入力PDFを取得"""
        root = tk.Tk()
        root.withdraw()
        return filedialog.askopenfilename(
            title="処理するPDFを選択",
            filetypes=[("PDFファイル", "*.pdf")],
            multiple=False
        )
    
    @staticmethod
    def get_output_file() -> Optional[str]:
        """提案されたファイル名付きで出力先を取得"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        suggested_name = f"ocr_enhanced_{timestamp}.pdf"
        return filedialog.asksaveasfilename(
            title="強化されたPDFを保存",
            defaultextension=".pdf",
            initialfile=suggested_name,
            filetypes=[("PDFファイル", "*.pdf")]
        )
    
    @staticmethod
    def get_processing_mode() -> Tuple[bool, dict]:
        """高度なオプション付きで処理の優先設定を取得"""
        root = tk.Tk()
        root.withdraw()
        
        use_japanese = messagebox.askyesno(
            "処理モード",
            "日本語テキストの最適化を使用しますか？"
        )
        
        # 追加の処理オプション
        options = {
            'parallel_processing': True,  # デフォルトではTrue
            'confidence_threshold': 60.0,
            'enhance_contrast': True,
            'remove_noise': True
        }
        
        return use_japanese, options

def main():
    """強化されたメインプログラムの実行"""
    ui = UserInterface()
    
    try:
        # 入力ファイルを取得
        input_pdf_path = ui.get_input_file()
        if not input_pdf_path:
            return
        
        # 出力ファイルを取得
        output_pdf_path = ui.get_output_file()
        if not output_pdf_path:
            return
        
        # 処理の優先設定を取得
        use_japanese_word_count, options = ui.get_processing_mode()
        
        # プロセッサーを初期化して実行
        processor = PDFProcessor()
        processor.process_pdf(input_pdf_path, output_pdf_path, use_japanese_word_count)
        
        # 統計情報を含む成功メッセージを表示
        messagebox.showinfo(
            "処理完了",
            "PDF処理が正常に完了しました！\n"
            f"出力先： {output_pdf_path}"
        )
        
    except Exception as e:
        logging.error(f"メイン実行でエラーが発生: {str(e)}")
        messagebox.showerror(
            "エラー",
            f"処理中にエラーが発生しました：\n{str(e)}\n"
            "詳細はログファイルを確認してください。"
        )

if __name__ == "__main__":
    main()

