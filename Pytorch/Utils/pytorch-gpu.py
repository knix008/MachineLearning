import torch
import tkinter as tk
from tkinter import ttk, scrolledtext
import threading
import time

class PyTorchGPUChecker:
    def __init__(self, root):
        self.root = root
        self.root.title("PyTorch GPU 확인 도구")
        self.root.geometry("600x500")
        self.root.configure(bg='#f0f0f0')
        
        # 메인 프레임
        main_frame = ttk.Frame(root, padding="20")
        main_frame.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # 제목
        title_label = ttk.Label(main_frame, text="PyTorch GPU 사용 가능 여부 확인", 
                               font=('Arial', 16, 'bold'))
        title_label.grid(row=0, column=0, columnspan=2, pady=(0, 20))
        
        # GPU 상태 표시
        self.status_frame = ttk.LabelFrame(main_frame, text="GPU 상태", padding="10")
        self.status_frame.grid(row=1, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 20))
        
        self.gpu_status_label = ttk.Label(self.status_frame, text="확인 중...", 
                                         font=('Arial', 12))
        self.gpu_status_label.grid(row=0, column=0, sticky=tk.W)
        
        # 세부 정보 표시
        self.details_frame = ttk.LabelFrame(main_frame, text="세부 정보", padding="10")
        self.details_frame.grid(row=2, column=0, columnspan=2, sticky=(tk.W, tk.E), pady=(0, 20))
        
        self.details_text = scrolledtext.ScrolledText(self.details_frame, height=15, width=70)
        self.details_text.grid(row=0, column=0, sticky=(tk.W, tk.E, tk.N, tk.S))
        
        # 버튼 프레임
        button_frame = ttk.Frame(main_frame)
        button_frame.grid(row=3, column=0, columnspan=2, pady=(0, 10))
        
        self.check_button = ttk.Button(button_frame, text="GPU 상태 확인", 
                                      command=self.check_gpu_status)
        self.check_button.grid(row=0, column=0, padx=(0, 10))
        
        self.tensor_test_button = ttk.Button(button_frame, text="텐서 테스트", 
                                           command=self.test_tensor_operations)
        self.tensor_test_button.grid(row=0, column=1, padx=(0, 10))
        
        self.clear_button = ttk.Button(button_frame, text="결과 지우기", 
                                     command=self.clear_results)
        self.clear_button.grid(row=0, column=2)
        
        # 그리드 가중치 설정
        root.columnconfigure(0, weight=1)
        root.rowconfigure(0, weight=1)
        main_frame.columnconfigure(0, weight=1)
        main_frame.rowconfigure(2, weight=1)
        self.details_frame.columnconfigure(0, weight=1)
        self.details_frame.rowconfigure(0, weight=1)
        
        # 초기 상태 확인
        self.check_gpu_status()
    
    def check_gpu_status(self):
        """GPU 상태를 확인하고 결과를 표시"""
        def check_in_thread():
            try:
                self.update_status("확인 중...")
                self.add_to_details("=== PyTorch GPU 상태 확인 ===\n")
                
                # CUDA 사용 가능 여부
                cuda_available = torch.cuda.is_available()
                self.add_to_details(f"CUDA 사용 가능: {cuda_available}\n")
                
                if cuda_available:
                    self.update_status("GPU 사용 가능 ✓", "green")
                    
                    # GPU 개수
                    gpu_count = torch.cuda.device_count()
                    self.add_to_details(f"GPU 개수: {gpu_count}\n")
                    
                    # 각 GPU 정보
                    for i in range(gpu_count):
                        gpu_name = torch.cuda.get_device_name(i)
                        gpu_memory = torch.cuda.get_device_properties(i).total_memory
                        gpu_memory_gb = gpu_memory / (1024**3)
                        
                        self.add_to_details(f"\nGPU {i}:\n")
                        self.add_to_details(f"  - 이름: {gpu_name}\n")
                        self.add_to_details(f"  - 메모리: {gpu_memory_gb:.2f} GB\n")
                        
                        # 메모리 사용량
                        if torch.cuda.is_available():
                            memory_allocated = torch.cuda.memory_allocated(i)
                            memory_reserved = torch.cuda.memory_reserved(i)
                            self.add_to_details(f"  - 할당된 메모리: {memory_allocated / (1024**2):.2f} MB\n")
                            self.add_to_details(f"  - 예약된 메모리: {memory_reserved / (1024**2):.2f} MB\n")
                    
                    # 현재 디바이스
                    current_device = torch.cuda.current_device()
                    self.add_to_details(f"\n현재 디바이스: {current_device}\n")
                    
                    # CUDA 버전
                    cuda_version = torch.version.cuda
                    self.add_to_details(f"CUDA 버전: {cuda_version}\n")
                    
                else:
                    self.update_status("GPU 사용 불가 ✗", "red")
                    self.add_to_details("GPU를 사용할 수 없습니다.\n")
                    self.add_to_details("가능한 원인:\n")
                    self.add_to_details("1. NVIDIA GPU가 설치되어 있지 않음\n")
                    self.add_to_details("2. CUDA 드라이버가 설치되어 있지 않음\n")
                    self.add_to_details("3. PyTorch가 CPU 버전으로 설치됨\n")
                
                # PyTorch 버전 정보
                pytorch_version = torch.__version__
                self.add_to_details(f"\nPyTorch 버전: {pytorch_version}\n")
                
                # 백엔드 정보
                self.add_to_details(f"cuDNN 사용 가능: {torch.backends.cudnn.enabled}\n")
                if torch.backends.cudnn.enabled:
                    self.add_to_details(f"cuDNN 버전: {torch.backends.cudnn.version()}\n")
                
                self.add_to_details("\n=== 확인 완료 ===\n\n")
                
            except Exception as e:
                self.update_status("오류 발생", "red")
                self.add_to_details(f"오류 발생: {str(e)}\n")
        
        threading.Thread(target=check_in_thread, daemon=True).start()
    
    def test_tensor_operations(self):
        """텐서 연산 테스트"""
        def test_in_thread():
            try:
                self.add_to_details("=== 텐서 연산 테스트 ===\n")
                
                # CPU 텐서 생성
                cpu_tensor = torch.rand(5, 3)
                self.add_to_details(f"CPU 텐서 생성:\n{cpu_tensor}\n")
                self.add_to_details(f"텐서 디바이스: {cpu_tensor.device}\n")
                
                if torch.cuda.is_available():
                    # GPU 텐서 생성
                    gpu_tensor = torch.rand(5, 3).cuda()
                    self.add_to_details(f"\nGPU 텐서 생성:\n{gpu_tensor}\n")
                    self.add_to_details(f"텐서 디바이스: {gpu_tensor.device}\n")
                    
                    # 간단한 연산 테스트
                    start_time = time.time()
                    result_cpu = torch.matmul(cpu_tensor, cpu_tensor.t())
                    cpu_time = time.time() - start_time
                    
                    start_time = time.time()
                    result_gpu = torch.matmul(gpu_tensor, gpu_tensor.t())
                    gpu_time = time.time() - start_time
                    
                    self.add_to_details(f"\n행렬 곱셈 연산 시간 비교:\n")
                    self.add_to_details(f"CPU: {cpu_time*1000:.4f} ms\n")
                    self.add_to_details(f"GPU: {gpu_time*1000:.4f} ms\n")
                    
                    # 더 큰 텐서로 성능 테스트
                    large_cpu = torch.rand(1000, 1000)
                    large_gpu = torch.rand(1000, 1000).cuda()
                    
                    start_time = time.time()
                    torch.matmul(large_cpu, large_cpu.t())
                    cpu_time_large = time.time() - start_time
                    
                    start_time = time.time()
                    torch.matmul(large_gpu, large_gpu.t())
                    torch.cuda.synchronize()  # GPU 연산 완료 대기
                    gpu_time_large = time.time() - start_time
                    
                    self.add_to_details(f"\n큰 행렬(1000x1000) 곱셈 시간 비교:\n")
                    self.add_to_details(f"CPU: {cpu_time_large*1000:.4f} ms\n")
                    self.add_to_details(f"GPU: {gpu_time_large*1000:.4f} ms\n")
                    
                    if cpu_time_large > gpu_time_large:
                        speedup = cpu_time_large / gpu_time_large
                        self.add_to_details(f"GPU 속도 향상: {speedup:.2f}x\n")
                
                self.add_to_details("\n=== 테스트 완료 ===\n\n")
                
            except Exception as e:
                self.add_to_details(f"테스트 오류: {str(e)}\n")
        
        threading.Thread(target=test_in_thread, daemon=True).start()
    
    def update_status(self, message, color="black"):
        """상태 메시지 업데이트"""
        self.gpu_status_label.config(text=message, foreground=color)
    
    def add_to_details(self, text):
        """세부 정보에 텍스트 추가"""
        self.details_text.insert(tk.END, text)
        self.details_text.see(tk.END)
        self.root.update_idletasks()
    
    def clear_results(self):
        """결과 지우기"""
        self.details_text.delete(1.0, tk.END)
        self.update_status("확인 준비")

def main():
    root = tk.Tk()
    app = PyTorchGPUChecker(root)
    root.mainloop()

if __name__ == "__main__":
    main()