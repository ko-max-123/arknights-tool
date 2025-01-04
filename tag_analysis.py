from pywinauto import Application
import pandas as pd
import tkinter as tk
from tkinter import messagebox
from PIL import ImageGrab, Image
import datetime
import pytesseract
import cv2
import numpy as np
import re
from PIL import Image
Image.MAX_IMAGE_PIXELS = None  # ピクセル制限解除

# Tesseractのパスを設定
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# タグリスト
tags = ["先鋒タイプ", "前衛タイプ", "狙撃タイプ", "重装タイプ", "医療タイプ", "補助タイプ", "術師タイプ", "特殊タイプ", "近距離", "遠距離", "火力", "防御", "COST回復", "範囲攻撃", "生存", "治療", "支援", "弱化", "減速", "強制移動", "牽制", "爆発力", "召喚", "高速再配置", "初期", "ロボット", "元素", "エリート", "上級エリート"]

# グローバル変数
global screenshot
screenshot = None

# 募集条件部分を切り抜き (3段中2段目を切り抜く)
def crop_recruitment_area(image):
    height, width = image.shape[:2]
    cropped_img = image[height // 3 : 2 * height // 3, :]  # 高さの2段目部分を切り抜く
    cv2.imwrite("cropped_image.png", cropped_img)  # 一時ファイルとして保存
    return cropped_img

# 解像度調整処理
def adjust_resolution(image, target_dpi=300):
    scale_factor = target_dpi / 96
    width = int(image.shape[1] * scale_factor)
    height = int(image.shape[0] * scale_factor)
    return cv2.resize(image, (width, height))

# テンプレートマッチングによる画像検索
def find_template_in_image(template_path, target_image):
    template = cv2.imread(template_path, 0)
    target_image_gray = cv2.cvtColor(target_image, cv2.COLOR_BGR2GRAY)  # 明示的にグレースケールに変換
    res = cv2.matchTemplate(target_image_gray, template, cv2.TM_CCOEFF_NORMED)
    min_val, max_val, min_loc, max_loc = cv2.minMaxLoc(res)
    return max_val > 0.9  # 類似度80%以上を検出基準とする

# 画像前処理関数
def preprocess_image(image_path):
    img = cv2.imread(image_path)
    img = crop_recruitment_area(img)

    # 画像サイズを4倍に拡大
    img = cv2.resize(img, None, fx=4, fy=4, interpolation=cv2.INTER_CUBIC)

    img = adjust_resolution(img)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.medianBlur(img, 3)

    # ネガポジ反転
    img = cv2.bitwise_not(img)

    # 二値化処理
    _, img = cv2.threshold(img, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)

    # コントラスト調整
    alpha, beta = 1.5, 10
    img = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)

    # シャープニング
    kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
    img = cv2.filter2D(img, -1, kernel)

    cv2.imwrite("processed_image.png", img)
    return img  # NumPy配列の形式で返す

# 画像前処理関数
def find_template_image(image_path):
    img = cv2.imread(image_path)
    img = crop_recruitment_area(img)

    cv2.imwrite("processed_image.png", img)
    return img  # NumPy配列の形式で返す


# アークナイツウィンドウキャプチャ
from pywinauto import Desktop

def capture_window():
    global screenshot
    try:
        app = Desktop(backend="uia").window(title_re=".*アークナイツ.*")
        rect = app.rectangle()
        screenshot = ImageGrab.grab(bbox=(rect.left, rect.top, rect.right, rect.bottom))
        screenshot.save("captured_image.png")
        messagebox.showinfo("キャプチャ", "アークナイツのウィンドウがキャプチャされました。")
    except Exception as e:
        messagebox.showerror("エラー", f"アークナイツのウィンドウが見つかりません: {e}")

# テキストレポートと画像の保存
def save_results(df, extracted_text):
    global screenshot
    timestamp = datetime.datetime.now().strftime("%m%d%H%M")
    txt_path = f"report_{timestamp}.txt"
    img_path = f"report_{timestamp}.png"
    with open(txt_path, 'w', encoding='utf-8') as file:
        file.write("募集条件解析結果\n\n")
        for index, row in df.iterrows():
            file.write(f"{row['タグ']}: {row['解析結果']}\n")
        file.write("\nOCR解析結果:\n")
        file.write(extracted_text)
    if screenshot:
        screenshot.save(img_path)
    messagebox.showinfo("完了", f"解析結果が保存されました。\nテキスト: {txt_path}\n画像: {img_path}")

# OCR解析処理
def analyze_image():
    global screenshot
    if screenshot is None:
        messagebox.showerror("エラー", "キャプチャされた画像がありません。")
        return []
    processed_img_path = "processed_image.png"
    processed_img = find_template_image("captured_image.png")
    extracted_text = pytesseract.image_to_string(
        Image.open(processed_img_path), lang='jpn', config='--psm 6 --oem 1 -c preserve_interword_spaces=1'
    )

    # テンプレートマッチングで画像内検出
    if find_template_in_image("tag_img/zenei.png", processed_img):
        extracted_text += " 前衛タイプ"
    if find_template_in_image("tag_img/jyusou.png", processed_img):
        extracted_text += " 重装タイプ"
    if find_template_in_image("tag_img/hojyo.png", processed_img):
        extracted_text += " 補助タイプ"
    if find_template_in_image("tag_img/sogeki.png", processed_img):
        extracted_text += " 狙撃タイプ"
    if find_template_in_image("tag_img/senpou.png", processed_img):
        extracted_text += " 先鋒タイプ"
    if find_template_in_image("tag_img/iryo.png", processed_img):
        extracted_text += " 医療タイプ"
    if find_template_in_image("tag_img/jyutushi.png", processed_img):
        extracted_text += " 術師タイプ"
    if find_template_in_image("tag_img/enkyori.png", processed_img):
        extracted_text += " 遠距離"
    if find_template_in_image("tag_img/kinkyori.png", processed_img):
        extracted_text += " 近距離"
    if find_template_in_image("tag_img/cost.png", processed_img):
        extracted_text += " COST回復"
    if find_template_in_image("tag_img/bougyo.png", processed_img):
        extracted_text += " 防御"
    if find_template_in_image("tag_img/shoki.png", processed_img):
        extracted_text += " 初期"
    if find_template_in_image("tag_img/karyoku.png", processed_img):
        extracted_text += " 火力"
    if find_template_in_image("tag_img/seizon.png", processed_img):
        extracted_text += " 生存"
    if find_template_in_image("tag_img/hani.png", processed_img):
        extracted_text += " 範囲攻撃"

    matched_tags = [tag for tag in tags if tag in extracted_text]
    return matched_tags, extracted_text

# 解析開始処理
def start_analysis():
    try:
        if screenshot is None:
            messagebox.showerror("エラー", "キャプチャされた画像がありません。")
            return
        extracted_tags, extracted_text = analyze_image()
        data = {
            "タグ": tags,
            "解析結果": ["取得済み" if tag in extracted_tags else "未取得" for tag in tags]
        }
        df = pd.DataFrame(data)
        save_results(df, extracted_text)
    except Exception as e:
        messagebox.showerror("エラー", str(e))

# GUIの構築
root = tk.Tk()
root.title("募集条件解析ツール")
root.geometry("300x200")

btn_capture = tk.Button(root, text="画面キャプチャ開始", command=capture_window)
btn_capture.pack(pady=10)

btn_start_analysis = tk.Button(root, text="解析開始", command=start_analysis)
btn_start_analysis.pack(pady=10)

root.mainloop()
