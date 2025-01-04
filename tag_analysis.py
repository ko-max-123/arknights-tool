from pywinauto import Application
import pandas as pd
import tkinter as tk
from tkinter import messagebox
from PIL import ImageGrab, Image
import datetime
import pytesseract
import cv2
import numpy as np
from PIL import Image
Image.MAX_IMAGE_PIXELS = None  # ピクセル制限解除

# Tesseractのパスを設定
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# タグリスト
tags = ["先鋒タイプ", "前衛タイプ", "狙撃タイプ", "重装タイプ", "医療タイプ", "補助タイプ", "術師タイプ", "特殊タイプ", "近距離", "遠距離", "火力", "防御", "COST回復", "範囲攻撃", "生存", "治療", "支援", "弱化", "減速", "強制移動", "牽制", "爆発力", "召喚", "高速再配置", "初期", "ロボット", "元素", "エリート", "上級エリート"]

# グローバル変数
global screenshot
screenshot = None

# 解像度調整処理
def adjust_resolution(image, target_dpi=300):
    scale_factor = target_dpi / 96
    width = int(image.shape[1] * scale_factor)
    height = int(image.shape[0] * scale_factor)
    return cv2.resize(image, (width, height))

# 画像前処理関数
def preprocess_image(image_path):
    img = cv2.imread(image_path)
    img = adjust_resolution(img)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img = cv2.medianBlur(img, 3)
    alpha, beta = 1.5, 10
    img = cv2.convertScaleAbs(img, alpha=alpha, beta=beta)
    kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
    img = cv2.filter2D(img, -1, kernel)
    cv2.imwrite("processed_image.png", img)
    return img

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
    preprocess_image("captured_image.png")
    extracted_text = pytesseract.image_to_string(
        Image.open(processed_img_path), lang='jpn', config='--psm 6 --oem 1 -c preserve_interword_spaces=1'
    )
    extracted_text = ''.join(filter(str.isprintable, extracted_text))
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
