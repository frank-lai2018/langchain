# uv 虛擬環境常用指令筆記

> 本專案的 venv 位置:`f:\langchain\langchain1_2\langchain1.2\`
> base python(另一個,勿混用):`D:\python\3.13.13\`

---

## 0. 重要觀念

- uv 建的虛擬環境**預設不裝 pip**,這是正常的,套件管理全部交給 uv。
- 用 `uv pip install`(不是傳統 `pip`)來裝套件,環境裡不需要真的有 pip。
- ⚠️ **同名陷阱**:操作前先 `activate`,確認提示字元有 `(langchain1.2)`,否則指令可能跑到 base python 去。

---

## 1. 進入專案 / 啟用環境

```powershell
# 進到專案資料夾
cd f:\langchain\langchain1_2

# 啟用虛擬環境(成功後提示字元前會出現 (langchain1.2))
.\langchain1.2\Scripts\activate

# 離開虛擬環境
deactivate
```

---

## 2. 安裝套件

```powershell
# 裝最新版
uv pip install langchain

# 指定版本(剛好這個版本)
uv pip install langchain==1.2.12

# 這個版本以上(有 > < 記得用雙引號)
uv pip install "langchain>=1.2.12"

# 某範圍內最新
uv pip install "langchain>=1.2,<1.3"

# 只鎖大版本、拿該系列最新
uv pip install "langchain==1.2.*"

# 一次裝多個
uv pip install langchain langchain-community chromadb
```

> 💡 小技巧:想知道有哪些版本可裝,故意打一個不存在的版本,uv 會在錯誤訊息列出所有可用版本:
> ```powershell
> uv pip install "langchain==999"
> ```

---

## 3. 查看已裝套件

```powershell
# 列出全部套件與版本
uv pip list

# 看某個套件的詳細資訊(含依賴)
uv pip show langchain

# 看依賴樹(誰依賴誰)
uv pip tree

# 匯出成 requirements 格式
uv pip freeze

# 不 activate 也能查:直接指定這個 venv 的 python
uv pip list --python f:\langchain\langchain1_2\langchain1.2\Scripts\python.exe
```

---

## 4. 更新 / 移除套件

```powershell
# 升級到最新版
uv pip install --upgrade langchain

# 移除套件
uv pip uninstall langchain
```

---

## 5. requirements 檔操作

```powershell
# 把目前環境套件匯出到檔案
uv pip freeze > requirements.txt

# 依 requirements 檔一次裝好
uv pip install -r requirements.txt
```

---

## 6. 建立新的虛擬環境(備忘)

```powershell
# 在目前資料夾建立名為 myenv 的 venv
uv venv myenv

# 指定 python 版本建立
uv venv myenv --python 3.13
```

---

## 7. 執行程式

```powershell
# 啟用環境後直接跑
python test.py

# 或不 activate,直接用該 venv 的 python 跑
.\langchain1.2\Scripts\python.exe test.py
```
