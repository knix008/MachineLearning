import tkinter as tk
from tkinter import ttk, messagebox
import datetime

class SimpleApp:
    def __init__(self, root):
        self.root = root
        self.root.title("간단한 QT 애플리케이션")
        self.root.geometry("400x500")
        self.root.configure(bg='#f0f0f0')
        
        # 스타일 설정
        style = ttk.Style()
        style.theme_use('clam')
        
        self.create_widgets()
        
    def create_widgets(self):
        # 메인 프레임
        main_frame = ttk.Frame(self.root, padding="20")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # 제목
        title_label = ttk.Label(main_frame, text="간단한 애플리케이션", 
                               font=('Arial', 16, 'bold'))
        title_label.grid(row=0, column=0, columnspan=2, pady=(0, 20))
        
        # 이름 입력
        ttk.Label(main_frame, text="이름:").grid(row=1, column=0, sticky=tk.W, pady=5)
        self.name_entry = ttk.Entry(main_frame, width=30)
        self.name_entry.grid(row=1, column=1, pady=5, padx=(10, 0))
        
        # 나이 입력
        ttk.Label(main_frame, text="나이:").grid(row=2, column=0, sticky=tk.W, pady=5)
        self.age_entry = ttk.Entry(main_frame, width=30)
        self.age_entry.grid(row=2, column=1, pady=5, padx=(10, 0))
        
        # 성별 선택
        ttk.Label(main_frame, text="성별:").grid(row=3, column=0, sticky=tk.W, pady=5)
        self.gender_var = tk.StringVar(value="남성")
        gender_frame = ttk.Frame(main_frame)
        gender_frame.grid(row=3, column=1, sticky=tk.W, pady=5, padx=(10, 0))
        
        ttk.Radiobutton(gender_frame, text="남성", variable=self.gender_var, 
                       value="남성").pack(side=tk.LEFT, padx=(0, 10))
        ttk.Radiobutton(gender_frame, text="여성", variable=self.gender_var, 
                       value="여성").pack(side=tk.LEFT)
        
        # 취미 선택
        ttk.Label(main_frame, text="취미:").grid(row=4, column=0, sticky=tk.W, pady=5)
        self.hobbies = []
        hobby_frame = ttk.Frame(main_frame)
        hobby_frame.grid(row=4, column=1, sticky=tk.W, pady=5, padx=(10, 0))
        
        hobbies_list = ["독서", "운동", "음악", "영화", "여행", "요리"]
        for i, hobby in enumerate(hobbies_list):
            var = tk.BooleanVar()
            self.hobbies.append((hobby, var))
            ttk.Checkbutton(hobby_frame, text=hobby, variable=var).pack(anchor=tk.W)
        
        # 버튼들
        button_frame = ttk.Frame(main_frame)
        button_frame.grid(row=5, column=0, columnspan=2, pady=20)
        
        ttk.Button(button_frame, text="정보 저장", 
                  command=self.save_info).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="시간 확인", 
                  command=self.show_time).pack(side=tk.LEFT, padx=5)
        ttk.Button(button_frame, text="초기화", 
                  command=self.clear_form).pack(side=tk.LEFT, padx=5)
        
        # 결과 표시 영역
        ttk.Label(main_frame, text="결과:").grid(row=6, column=0, sticky=tk.W, pady=(20, 5))
        self.result_text = tk.Text(main_frame, height=8, width=40)
        self.result_text.grid(row=7, column=0, columnspan=2, pady=5)
        
        # 스크롤바
        scrollbar = ttk.Scrollbar(main_frame, orient=tk.VERTICAL, command=self.result_text.yview)
        scrollbar.grid(row=7, column=2, sticky=(tk.N, tk.S))
        self.result_text.configure(yscrollcommand=scrollbar.set)
        
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
        
        # 선택된 취미들
        selected_hobbies = [hobby for hobby, var in self.hobbies if var.get()]
        
        # 결과 생성
        result = f"=== 저장된 정보 ===\n"
        result += f"이름: {name}\n"
        result += f"나이: {age}세\n"
        result += f"성별: {gender}\n"
        result += f"취미: {', '.join(selected_hobbies) if selected_hobbies else '없음'}\n"
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
        
        # 취미 체크박스 초기화
        for hobby, var in self.hobbies:
            var.set(False)
        
        # 결과 텍스트 초기화
        self.result_text.delete(1.0, tk.END)
        
        messagebox.showinfo("초기화", "모든 입력이 초기화되었습니다.")

def main():
    root = tk.Tk()
    app = SimpleApp(root)
    root.mainloop()

if __name__ == "__main__":
    main() 