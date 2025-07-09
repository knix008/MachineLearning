import tkinter as tk
from tkinter import messagebox
import datetime

class BasicApp:
    def __init__(self):
        self.root = tk.Tk()
        self.root.title("기본 GUI 애플리케이션")
        self.root.geometry("350x400")
        self.root.configure(bg='lightblue')
        
        self.create_widgets()
        
    def create_widgets(self):
        # 제목
        title = tk.Label(self.root, text="간단한 GUI 애플리케이션", 
                        font=('Arial', 16, 'bold'), bg='lightblue')
        title.pack(pady=20)
        
        # 프레임 생성
        main_frame = tk.Frame(self.root, bg='lightblue')
        main_frame.pack(pady=10)
        
        # 이름 입력
        tk.Label(main_frame, text="이름:", bg='lightblue', font=('Arial', 12)).pack(anchor='w')
        self.name_entry = tk.Entry(main_frame, width=25, font=('Arial', 12))
        self.name_entry.pack(pady=5)
        
        # 나이 입력
        tk.Label(main_frame, text="나이:", bg='lightblue', font=('Arial', 12)).pack(anchor='w', pady=(10, 0))
        self.age_entry = tk.Entry(main_frame, width=25, font=('Arial', 12))
        self.age_entry.pack(pady=5)
        
        # 성별 선택
        tk.Label(main_frame, text="성별:", bg='lightblue', font=('Arial', 12)).pack(anchor='w', pady=(10, 0))
        self.gender_var = tk.StringVar(value="남성")
        gender_frame = tk.Frame(main_frame, bg='lightblue')
        gender_frame.pack()
        
        tk.Radiobutton(gender_frame, text="남성", variable=self.gender_var, 
                      value="남성", bg='lightblue', font=('Arial', 11)).pack(side='left', padx=10)
        tk.Radiobutton(gender_frame, text="여성", variable=self.gender_var, 
                      value="여성", bg='lightblue', font=('Arial', 11)).pack(side='left', padx=10)
        
        # 버튼들
        button_frame = tk.Frame(self.root, bg='lightblue')
        button_frame.pack(pady=20)
        
        tk.Button(button_frame, text="정보 저장", command=self.save_info,
                 bg='green', fg='white', font=('Arial', 11), width=10).pack(side='left', padx=5)
        tk.Button(button_frame, text="시간 확인", command=self.show_time,
                 bg='blue', fg='white', font=('Arial', 11), width=10).pack(side='left', padx=5)
        tk.Button(button_frame, text="초기화", command=self.clear_form,
                 bg='red', fg='white', font=('Arial', 11), width=10).pack(side='left', padx=5)
        
        # 결과 표시 영역
        tk.Label(self.root, text="결과:", bg='lightblue', font=('Arial', 12, 'bold')).pack(anchor='w', padx=20, pady=(20, 5))
        
        # 텍스트 영역과 스크롤바
        text_frame = tk.Frame(self.root)
        text_frame.pack(padx=20, fill='both', expand=True)
        
        self.result_text = tk.Text(text_frame, height=8, width=40, font=('Arial', 10))
        scrollbar = tk.Scrollbar(text_frame, command=self.result_text.yview)
        self.result_text.configure(yscrollcommand=scrollbar.set)
        
        self.result_text.pack(side='left', fill='both', expand=True)
        scrollbar.pack(side='right', fill='y')
        
    def save_info(self):
        name = self.name_entry.get().strip()
        age = self.age_entry.get().strip()
        gender = self.gender_var.get()
        
        if not name or not age:
            messagebox.showwarning("경고", "이름과 나이를 모두 입력해주세요.")
            return
        
        try:
            age_int = int(age)
            if age_int <= 0 or age_int > 150:
                raise ValueError
        except ValueError:
            messagebox.showerror("오류", "올바른 나이를 입력해주세요.")
            return
        
        # 결과 생성
        result = f"=== 저장된 정보 ===\n"
        result += f"이름: {name}\n"
        result += f"나이: {age}세\n"
        result += f"성별: {gender}\n"
        result += f"저장 시간: {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        result += "=" * 20 + "\n\n"
        
        # 결과 텍스트에 추가
        self.result_text.insert(tk.END, result)
        self.result_text.see(tk.END)
        
        messagebox.showinfo("성공", "정보가 성공적으로 저장되었습니다!")
        
    def show_time(self):
        current_time = datetime.datetime.now().strftime('%Y년 %m월 %d일 %H시 %M분 %S초')
        time_message = f"현재 시간: {current_time}\n\n"
        self.result_text.insert(tk.END, time_message)
        self.result_text.see(tk.END)
        
    def clear_form(self):
        # 입력 필드 초기화
        self.name_entry.delete(0, tk.END)
        self.age_entry.delete(0, tk.END)
        self.gender_var.set("남성")
        
        # 결과 텍스트 초기화
        self.result_text.delete(1.0, tk.END)
        
        messagebox.showinfo("초기화", "모든 입력이 초기화되었습니다.")
    
    def run(self):
        self.root.mainloop()

def main():
    app = BasicApp()
    app.run()

if __name__ == "__main__":
    main() 