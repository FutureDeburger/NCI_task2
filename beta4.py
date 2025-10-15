import tkinter as tk
from tkinter import filedialog, messagebox, ttk
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import pywt
import os
import scipy.io
import pandas as pd


class WaveletZVPAnalyzer:
    def __init__(self, root):
        self.root = root
        self.root.title("Вейвлет-анализ Зрительных Вызванных Потенциалов (ЗВП)")
        self.root.geometry("1600x1200")

        self.data = None
        self.channel_names = []
        self.sampling_rate = 1000  # Hz по умолчанию
        self.current_view = 'analysis'
        self.current_channel = 0
        self.wavelet_coefficients = None
        self.wavelet_frequencies = None

        # Фиксированные параметры анализа
        self.wavelet = 'cmor'  # Фиксированный вейвлет
        self.min_freq = 1  # Минимальная частота
        self.max_freq = 30  # Максимальная частота

        self.create_widgets()

    def create_widgets(self):
        # Основной контейнер с разделением на две части
        main_paned = tk.PanedWindow(self.root, orient=tk.HORIZONTAL, sashrelief=tk.RAISED, sashwidth=4)
        main_paned.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Левая панель - графики и управление
        left_frame = tk.Frame(main_paned)
        main_paned.add(left_frame, width=1000)

        # Правая панель - результаты
        right_frame = tk.Frame(main_paned)
        main_paned.add(right_frame, width=600)

        # ЗАГОЛОВОК
        title_label = tk.Label(left_frame, text="Вейвлет-анализ Зрительных Вызванных Потенциалов",
                               font=("Arial", 16, "bold"))
        title_label.pack(pady=15)

        # Фрейм для кнопок
        button_frame = tk.Frame(left_frame)
        button_frame.pack(pady=10)

        # Первая строка кнопок
        button_row1 = tk.Frame(button_frame)
        button_row1.pack(pady=5)

        # Кнопка загрузки файла
        load_btn = tk.Button(button_row1, text="ЗАГРУЗИТЬ ФАЙЛ .ASC",
                             command=self.load_file,
                             font=("Arial", 11, "bold"),
                             width=22,
                             height=2,
                             bg="#4CAF50",
                             fg="white",
                             activebackground="#45a049")
        load_btn.pack(side=tk.LEFT, padx=5)

        # Кнопка просмотра исходного сигнала
        self.view_signal_btn = tk.Button(button_row1, text="ПРОСМОТР ЭЭГ",
                                         command=self.view_raw_signal,
                                         font=("Arial", 11, "bold"),
                                         width=22,
                                         height=2,
                                         bg="#607D8B",
                                         fg="white",
                                         activebackground="#546E7A",
                                         state="disabled")
        self.view_signal_btn.pack(side=tk.LEFT, padx=5)

        # Кнопка вейвлет-анализа
        self.analyze_wavelet_btn = tk.Button(button_row1, text="ВЕЙВЛЕТ-АНАЛИЗ",
                                             command=self.perform_wavelet_analysis,
                                             font=("Arial", 11, "bold"),
                                             width=22,
                                             height=2,
                                             bg="#2196F3",
                                             fg="white",
                                             activebackground="#0b7dda",
                                             state="disabled")
        self.analyze_wavelet_btn.pack(side=tk.LEFT, padx=5)

        # Вторая строка кнопок
        button_row2 = tk.Frame(button_frame)
        button_row2.pack(pady=5)

        # Кнопка сохранения результатов
        self.save_btn = tk.Button(button_row2, text="СОХРАНИТЬ РЕЗУЛЬТАТЫ",
                                  command=self.save_results,
                                  font=("Arial", 11, "bold"),
                                  width=22,
                                  height=2,
                                  bg="#FF9800",
                                  fg="white",
                                  state="disabled")
        self.save_btn.pack(side=tk.LEFT, padx=5)

        # Кнопка возврата к анализу
        self.back_btn = tk.Button(button_row2, text="ВЕРНУТЬСЯ К АНАЛИЗУ",
                                  command=self.show_analysis_view,
                                  font=("Arial", 11, "bold"),
                                  width=22,
                                  height=2,
                                  bg="#795548",
                                  fg="white",
                                  state="disabled")
        self.back_btn.pack(side=tk.LEFT, padx=5)

        # Фрейм управления каналами для просмотра ЭЭГ
        self.channel_control_frame = tk.Frame(button_frame)

        # Фрейм управления параметрами анализа
        self.analysis_control_frame = tk.Frame(button_frame)

        # Выбор канала для анализа
        channel_analysis_frame = tk.Frame(self.analysis_control_frame)
        channel_analysis_frame.pack(pady=5)

        tk.Label(channel_analysis_frame, text="Канал для анализа:", font=("Arial", 10, "bold")).pack(side=tk.LEFT,
                                                                                                     padx=5)

        self.analysis_channel_combo = ttk.Combobox(channel_analysis_frame, state="readonly", width=15)
        self.analysis_channel_combo.pack(side=tk.LEFT, padx=5)

        # Информация о параметрах анализа
        params_frame = tk.Frame(self.analysis_control_frame)
        params_frame.pack(pady=5)

        params_label = tk.Label(params_frame,
                                text="Параметры анализа: Complex Morlet (cmor), частоты 1-30 Гц",
                                font=("Arial", 9),
                                fg="gray")
        params_label.pack()

        # Информация о файле
        self.file_label = tk.Label(left_frame, text="Файл не загружен",
                                   font=("Arial", 12))
        self.file_label.pack(pady=5)

        # Область для графиков в левой панели
        self.create_plot_area(left_frame)

        # Область для результатов в правой панели
        self.create_results_area(right_frame)

    def create_plot_area(self, parent):
        plot_frame = tk.Frame(parent)
        plot_frame.pack(fill='both', expand=True, padx=10, pady=10)

        self.fig = plt.figure(figsize=(12, 9))
        self.setup_analysis_grid()

        self.canvas = FigureCanvasTkAgg(self.fig, master=plot_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill='both', expand=True)

    def setup_analysis_grid(self):
        self.fig.clear()
        # Сетка для вейвлет-анализа: исходный сигнал и спектрограмма
        self.gs = plt.GridSpec(2, 1, figure=self.fig, height_ratios=[1, 1])

        self.ax_signal = self.fig.add_subplot(self.gs[0])
        self.ax_spectrogram = self.fig.add_subplot(self.gs[1])

        self.fig.tight_layout(pad=3.0)

    def setup_signal_grid(self):
        self.fig.clear()
        self.ax_signal_only = self.fig.add_subplot(111)
        self.fig.tight_layout(pad=3.0)

    def create_results_area(self, parent):
        results_frame = tk.Frame(parent)
        results_frame.pack(fill='both', expand=True, padx=10, pady=10)

        self.results_label = tk.Label(results_frame, text="РЕЗУЛЬТАТЫ АНАЛИЗА:",
                                      font=("Arial", 14, "bold"))
        self.results_label.pack(anchor='w', pady=(0, 10))

        # Создаем фрейм для текстовой области с фиксированной высотой
        text_frame = tk.Frame(results_frame)
        text_frame.pack(fill='both', expand=True)

        # Добавляем виджет Text с фиксированной высотой
        self.results_text = tk.Text(text_frame,
                                    height=35,
                                    width=80,
                                    font=("Courier", 12),
                                    wrap=tk.WORD,
                                    bg="#f8f9fa",
                                    relief=tk.SUNKEN,
                                    bd=2,
                                    padx=10,
                                    pady=10)
        self.results_text.pack(fill='both', expand=True)

        # Отключаем возможность изменения размера текстового поля
        self.results_text.config(state="normal")

        results_buttons_frame = tk.Frame(results_frame)
        results_buttons_frame.pack(fill=tk.X, pady=(10, 0))

        clear_btn = tk.Button(results_buttons_frame,
                              text="ОЧИСТИТЬ РЕЗУЛЬТАТЫ",
                              command=self.clear_results,
                              font=("Arial", 10, "bold"),
                              width=22,
                              height=1,
                              bg="#F44336",
                              fg="white")
        clear_btn.pack(side=tk.LEFT, padx=5)

        copy_btn = tk.Button(results_buttons_frame,
                             text="КОПИРОВАТЬ РЕЗУЛЬТАТЫ",
                             command=self.copy_results,
                             font=("Arial", 10, "bold"),
                             width=22,
                             height=1,
                             bg="#2196F3",
                             fg="white")
        copy_btn.pack(side=tk.LEFT, padx=5)

        scrollbar = tk.Scrollbar(self.results_text)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.results_text.config(yscrollcommand=scrollbar.set)
        scrollbar.config(command=self.results_text.yview)

        h_scrollbar = tk.Scrollbar(self.results_text, orient=tk.HORIZONTAL)
        h_scrollbar.pack(side=tk.BOTTOM, fill=tk.X)
        self.results_text.config(xscrollcommand=h_scrollbar.set)
        h_scrollbar.config(command=self.results_text.xview)

    def clear_results(self):
        self.results_text.delete(1.0, tk.END)

    def copy_results(self):
        results = self.results_text.get(1.0, tk.END)
        if results.strip():
            self.root.clipboard_clear()
            self.root.clipboard_append(results)
            messagebox.showinfo("Успех", "Результаты скопированы в буфер обмена!")
        else:
            messagebox.showwarning("Внимание", "Нет данных для копирования")

    def load_asc_file(self, file_path):
        """Загрузка ASC файла NeuroSoft"""
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                lines = file.readlines()

            # Парсим заголовок
            header_info = {}
            data_lines = []

            for line in lines:
                line = line.strip()
                if line.startswith(';'):
                    # Обрабатываем заголовок
                    if 'Frequency:' in line:
                        freq_str = line.split('Frequency:')[1].strip()
                        header_info['frequency'] = int(freq_str)
                    elif 'Derivations number:' in line:
                        deriv_str = line.split('Derivations number:')[1].strip()
                        header_info['derivations'] = int(deriv_str)
                    elif 'Derivation names:' in line:
                        names_str = line.split('Derivation names:')[1].strip()
                        self.channel_names = [name.strip() for name in names_str.split(',')]
                elif line and not line.startswith(';'):
                    # Данные
                    data_lines.append(line)

            # Парсим данные
            data = []
            for line in data_lines:
                if line.strip():
                    try:
                        row = [float(x) for x in line.split()]
                        data.append(row)
                    except ValueError:
                        continue

            self.data = np.array(data)

            # Устанавливаем частоту дискретизации из заголовка
            if 'frequency' in header_info:
                self.sampling_rate = header_info['frequency']
            else:
                self.sampling_rate = 1000  # значение по умолчанию

            return True

        except Exception as e:
            raise Exception(f"Ошибка чтения ASC файла: {str(e)}")

    def load_file(self):
        file_path = filedialog.askopenfilename(
            title="Выберите файл данных ЗВП",
            filetypes=[("ASC files", "*.asc"), ("MAT files", "*.mat"),
                       ("CSV files", "*.csv"), ("TXT files", "*.txt"),
                       ("All files", "*.*")]
        )

        if file_path:
            try:
                if file_path.endswith('.asc'):
                    # Загрузка ASC файла
                    self.load_asc_file(file_path)
                elif file_path.endswith('.mat'):
                    # Загрузка MAT файла
                    mat_data = scipy.io.loadmat(file_path)
                    self.data = self.extract_data_from_mat(mat_data)
                    self.channel_names = ['Channel_1']  # по умолчанию
                elif file_path.endswith('.csv'):
                    # Загрузка CSV файла
                    df = pd.read_csv(file_path)
                    self.data = df.values
                    if self.data.ndim == 1:
                        self.data = self.data.reshape(-1, 1)
                    self.channel_names = [f'Channel_{i + 1}' for i in range(self.data.shape[1])]
                elif file_path.endswith('.txt'):
                    # Загрузка текстового файла
                    self.data = np.loadtxt(file_path)
                    if self.data.ndim == 1:
                        self.data = self.data.reshape(-1, 1)
                    self.channel_names = [f'Channel_{i + 1}' for i in range(self.data.shape[1])]
                else:
                    # Попытка загрузить как текстовый файл
                    self.data = np.loadtxt(file_path)
                    if self.data.ndim == 1:
                        self.data = self.data.reshape(-1, 1)
                    self.channel_names = [f'Channel_{i + 1}' for i in range(self.data.shape[1])]

                self.total_duration = len(self.data) / self.sampling_rate

                # Обновляем интерфейс с информацией о каналах
                self.update_channel_interface()

                self.file_label.config(text=f"Загружен: {os.path.basename(file_path)}")
                self.analyze_wavelet_btn.config(state="normal")
                self.save_btn.config(state="normal")
                self.view_signal_btn.config(state="normal")
                self.back_btn.config(state="normal")

                # Активируем панель управления
                self.analysis_control_frame.pack(pady=10)

                messagebox.showinfo("Успех",
                                    f"Файл загружен!\nКаналов: {self.data.shape[1]}\nОтсчетов: {self.data.shape[0]}\nДлительность: {self.total_duration:.2f} сек\nЧастота: {self.sampling_rate} Гц")

            except Exception as e:
                messagebox.showerror("Ошибка", f"Не удалось загрузить файл:\n{str(e)}")

    def update_channel_interface(self):
        """Обновление интерфейса для работы с каналами"""
        # Очищаем старые кнопки каналов
        for widget in self.channel_control_frame.winfo_children():
            widget.destroy()

        # Создаем новые кнопки каналов
        channel_frame = tk.Frame(self.channel_control_frame)
        channel_frame.pack(pady=5)

        tk.Label(channel_frame, text="Каналы:", font=("Arial", 10, "bold")).pack(side=tk.LEFT, padx=5)

        self.channel_buttons = []
        for i, channel in enumerate(self.channel_names):
            btn = tk.Button(channel_frame, text=channel,
                            command=lambda idx=i: self.switch_channel(idx),
                            font=("Arial", 9, "bold"),
                            width=8,
                            height=1,
                            bg="#E0E0E0",
                            fg="black")
            btn.pack(side=tk.LEFT, padx=2)
            self.channel_buttons.append(btn)

        # Обновляем комбобокс для выбора канала анализа
        self.analysis_channel_combo['values'] = self.channel_names
        self.analysis_channel_combo.set(self.channel_names[0])

        # Активируем первую кнопку
        if self.channel_buttons:
            self.channel_buttons[0].config(bg="#2196F3", fg="white")

    def switch_channel(self, channel_idx):
        """Переключение канала для просмотра"""
        if self.data is None or self.current_view != 'signal':
            return

        self.current_channel = channel_idx

        for i, btn in enumerate(self.channel_buttons):
            if i == channel_idx:
                btn.config(bg="#2196F3", fg="white")
            else:
                btn.config(bg="#E0E0E0", fg="black")

        self.update_signal_display()

    def extract_data_from_mat(self, mat_data):
        """Извлечение данных из MAT файла"""
        for key in mat_data.keys():
            if not key.startswith('__') and isinstance(mat_data[key], np.ndarray):
                data = mat_data[key]
                if data.ndim == 1:
                    return data.reshape(-1, 1)
                elif data.ndim == 2:
                    return data
        raise ValueError("Не удалось найти подходящие данные в MAT файла")

    def view_raw_signal(self):
        if self.data is None:
            return

        try:
            self.current_view = 'signal'
            self.setup_signal_grid()
            self.channel_control_frame.pack(pady=10)
            self.update_signal_display()

        except Exception as e:
            messagebox.showerror("Ошибка", f"Не удалось отобразить сигнал:\n{str(e)}")

    def update_signal_display(self):
        if self.data is None or self.current_view != 'signal':
            return

        try:
            self.ax_signal_only.clear()

            channel_data = self.data[:, self.current_channel]
            time = np.arange(len(channel_data)) / self.sampling_rate

            self.ax_signal_only.plot(time, channel_data, color='#2196F3', linewidth=1)
            self.ax_signal_only.set_title(f'ЭЭГ - Канал {self.channel_names[self.current_channel]}',
                                          fontweight='bold', fontsize=14)
            self.ax_signal_only.set_ylabel('Амплитуда (мкВ)', fontsize=12)
            self.ax_signal_only.set_xlabel('Время (секунды)', fontsize=12)
            self.ax_signal_only.grid(True, alpha=0.3)

            y_margin = (np.max(channel_data) - np.min(channel_data)) * 0.1
            if y_margin == 0:
                y_margin = 1
            self.ax_signal_only.set_ylim(np.min(channel_data) - y_margin, np.max(channel_data) + y_margin)

            self.fig.tight_layout()
            self.canvas.draw()

            self.update_signal_info()

        except Exception as e:
            print(f"Ошибка при обновлении сигнала: {e}")

    def update_signal_info(self):
        self.results_text.delete(1.0, tk.END)

        channel_data = self.data[:, self.current_channel]

        info_text = f"ИНФОРМАЦИЯ О СИГНАЛЕ ЭЭГ - {self.channel_names[self.current_channel]}\n"
        info_text += "=" * 80 + "\n\n"
        info_text += f"ОБЩАЯ ИНФОРМАЦИЯ:\n"
        info_text += f"• Канал: {self.channel_names[self.current_channel]}\n"
        info_text += f"• Длительность записи: {self.total_duration:.2f} сек\n"
        info_text += f"• Частота дискретизации: {self.sampling_rate} Гц\n"
        info_text += f"• Всего отсчетов: {len(self.data)}\n"

        info_text += f"\nСТАТИСТИКА СИГНАЛА:\n"
        info_text += "-" * 50 + "\n"
        info_text += f"• Минимум: {np.min(channel_data):.2f} мкВ\n"
        info_text += f"• Максимум: {np.max(channel_data):.2f} мкВ\n"
        info_text += f"• Среднее: {np.mean(channel_data):.2f} мкВ\n"
        info_text += f"• Стандартное отклонение: {np.std(channel_data):.2f} мкВ\n"
        info_text += f"• Динамический диапазон: {np.max(channel_data) - np.min(channel_data):.2f} мкВ\n"

        info_text += f"\nВСЕ КАНАЛЫ:\n"
        info_text += "-" * 25 + "\n"
        for i, name in enumerate(self.channel_names):
            info_text += f"• {name}\n"

        self.results_text.insert(1.0, info_text)
        self.results_label.config(text=f"ИНФОРМАЦИЯ О СИГНАЛЕ - {self.channel_names[self.current_channel]}:")

    def show_analysis_view(self):
        if self.data is None:
            return

        self.current_view = 'analysis'
        self.setup_analysis_grid()
        self.results_label.config(text="РЕЗУЛЬТАТЫ АНАЛИЗА:")
        self.results_text.delete(1.0, tk.END)
        self.results_text.insert(1.0, "Готов к анализу. Нажмите 'ВЕЙВЛЕТ-АНАЛИЗ'.")
        self.channel_control_frame.pack_forget()
        self.canvas.draw()

    def get_current_analysis_channel(self):
        """Получение данных выбранного канала для анализа"""
        if self.analysis_channel_combo.get():
            channel_idx = self.channel_names.index(self.analysis_channel_combo.get())
            return self.data[:, channel_idx]
        else:
            return self.data[:, 0]  # по умолчанию первый канал

    def perform_wavelet_analysis(self):
        if self.data is None:
            return

        self.show_analysis_view()
        self.results_text.delete(1.0, tk.END)
        self.current_view = 'analysis'

        try:
            # Получаем данные выбранного канала
            channel_data = self.get_current_analysis_channel()

            # Нормализуем данные для лучшей стабильности
            channel_data = (channel_data - np.mean(channel_data)) / np.std(channel_data)

            # Генерируем шкалы для вейвлет-преобразования
            # Используем логарифмическую шкалу для лучшего покрытия частот
            num_scales = 100
            scales = np.logspace(np.log10(1), np.log10(128), num_scales)

            # Выполняем непрерывное вейвлет-преобразование с фиксированным cmor
            coefficients, frequencies = pywt.cwt(channel_data, scales, self.wavelet,
                                                 sampling_period=1.0 / self.sampling_rate)

            # Фильтруем частоты по фиксированному диапазону
            freq_mask = (frequencies >= self.min_freq) & (frequencies <= self.max_freq)
            coefficients = coefficients[freq_mask, :]
            frequencies = frequencies[freq_mask]

            # Сохраняем результаты для отображения
            self.wavelet_coefficients = coefficients
            self.wavelet_frequencies = frequencies

            # Строим графики
            self.plot_wavelet_results(coefficients, frequencies, channel_data)

            # Обновляем информацию о результатах
            self.update_wavelet_results_info(coefficients, frequencies)

        except Exception as e:
            messagebox.showerror("Ошибка анализа", f"Ошибка вейвлет-анализа:\n{str(e)}")
            import traceback
            print(traceback.format_exc())

    def plot_wavelet_results(self, coefficients, frequencies, channel_data):
        """Построение графиков результатов вейвлет-анализа"""
        # Очищаем графики
        self.ax_signal.clear()
        self.ax_spectrogram.clear()

        time = np.arange(len(channel_data)) / self.sampling_rate

        # 1. Исходный сигнал (полный)
        self.ax_signal.plot(time, channel_data, 'b-', linewidth=1)
        self.ax_signal.set_ylabel('Амплитуда (норм.)')
        self.ax_signal.set_title(f'Исходный сигнал ЗВП - {self.analysis_channel_combo.get()}')
        self.ax_signal.grid(True, alpha=0.3)
        self.ax_signal.set_xlim(0, self.total_duration)

        # 2. Вейвлет-спектрограмма (полная)
        if len(coefficients) > 0 and len(frequencies) > 0:
            im = self.ax_spectrogram.imshow(np.abs(coefficients),
                                            extent=[time[0], time[-1],
                                                    frequencies[-1], frequencies[0]],
                                            aspect='auto', cmap='jet',
                                            interpolation='bilinear')
            self.ax_spectrogram.set_ylabel('Частота (Гц)')
            self.ax_spectrogram.set_title('Вейвлет-спектрограмма (Complex Morlet)')
            self.ax_spectrogram.set_xlabel('Время (сек)')
            plt.colorbar(im, ax=self.ax_spectrogram, label='Амплитуда')

            # Добавляем сетку для лучшей читаемости
            self.ax_spectrogram.grid(True, alpha=0.3, linestyle='--')
        else:
            self.ax_spectrogram.text(0.5, 0.5, 'Нет данных для отображения',
                                     ha='center', va='center', transform=self.ax_spectrogram.transAxes)

        self.fig.tight_layout()
        self.canvas.draw()

    def update_wavelet_results_info(self, coefficients, frequencies):
        """Обновление информации о результатах вейвлет-анализа"""
        if len(coefficients) == 0:
            info_text = "Нет данных для отображения результатов вейвлет-анализа.\n"
            info_text += "Проверьте параметры анализа."
        else:
            max_coeff = np.max(np.abs(coefficients))
            mean_coeff = np.mean(np.abs(coefficients))

            # Находим частоту с максимальной энергией
            energy_by_freq = np.mean(np.abs(coefficients), axis=1)
            dominant_freq_idx = np.argmax(energy_by_freq)
            dominant_freq = frequencies[dominant_freq_idx]

            info_text = f"РЕЗУЛЬТАТЫ ВЕЙВЛЕТ-АНАЛИЗА ЗВП\n"
            info_text += "=" * 80 + "\n\n"
            info_text += f"ПАРАМЕТРЫ АНАЛИЗА:\n"
            info_text += f"• Канал: {self.analysis_channel_combo.get()}\n"
            info_text += f"• Вейвлет: Complex Morlet (cmor)\n"
            info_text += f"• Частотный диапазон: {self.min_freq}-{self.max_freq} Гц\n"
            info_text += f"• Частота дискретизации: {self.sampling_rate} Гц\n"
            info_text += f"• Количество шкал: {len(coefficients)}\n\n"

            info_text += f"РЕЗУЛЬТАТЫ АНАЛИЗА:\n"
            info_text += "-" * 50 + "\n"
            info_text += f"• Максимальная амплитуда коэффициентов: {max_coeff:.4f}\n"
            info_text += f"• Средняя амплитуда коэффициентов: {mean_coeff:.4f}\n"
            info_text += f"• Доминирующая частота: {dominant_freq:.2f} Гц\n"
            # info_text += f"• Размерность матрицы коэффициентов: {coefficients.shape}\n\n"

        info_text += f"ДОСТУПНЫЕ КАНАЛЫ:\n"
        info_text += "-" * 25 + "\n"
        for name in self.channel_names:
            info_text += f"• {name}\n"

        self.results_text.insert(1.0, info_text)
        self.results_label.config(text="РЕЗУЛЬТАТЫ ВЕЙВЛЕТ-АНАЛИЗА:")

    def save_results(self):
        file_path = filedialog.asksaveasfilename(
            title="Сохранить результаты",
            defaultextension=".txt",
            filetypes=[("Text files", "*.txt"), ("All files", "*.*")]
        )

        if file_path:
            try:
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(self.results_text.get(1.0, tk.END))

                plot_path = file_path.replace('.txt', '_plot.png')
                self.fig.savefig(plot_path, dpi=300, bbox_inches='tight')

                messagebox.showinfo("Успех", f"Результаты сохранены!\nТекст: {file_path}\nГрафик: {plot_path}")
            except Exception as e:
                messagebox.showerror("Ошибка", f"Не удалось сохранить:\n{str(e)}")


if __name__ == "__main__":
    root = tk.Tk()
    app = WaveletZVPAnalyzer(root)
    root.mainloop()