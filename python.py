        import streamlit as st
import pandas as pd
from google import genai
from google.genai.errors import APIError

# --- Cấu hình Trang Streamlit ---
st.set_page_config(
    page_title="App Phân Tích Báo Cáo Tài Chính",
    layout="wide"
)

st.title("Ứng dụng Phân Tích Báo cáo Tài Chính 📊 và Chatbot AI")

# --- Khởi tạo session state cho lịch sử chat (MANDATORY cho Chatbot) ---
if "chat_messages" not in st.session_state:
    st.session_state["chat_messages"] = [
        {"role": "assistant", "content": "Chào bạn! Vui lòng tải lên Báo cáo Tài chính (Excel). Sau khi phân tích xong, tôi có thể trả lời các câu hỏi của bạn về dữ liệu đó."}
    ]

# --- Hàm tính toán chính (Sử dụng Caching để Tối ưu hiệu suất) ---
@st.cache_data
def process_financial_data(df):
    """Thực hiện các phép tính Tăng trưởng và Tỷ trọng."""
    
    # Đảm bảo các giá trị là số để tính toán
    numeric_cols = ['Năm trước', 'Năm sau']
    for col in numeric_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0)
    
    # 1. Tính Tốc độ Tăng trưởng
    df['Tốc độ tăng trưởng (%)'] = (
        (df['Năm sau'] - df['Năm trước']) / df['Năm trước'].replace(0, 1e-9)
    ) * 100

    # 2. Tính Tỷ trọng theo Tổng Tài sản
    tong_tai_san_row = df[df['Chỉ tiêu'].str.contains('TỔNG CỘNG TÀI SẢN', case=False, na=False)]
    
    if tong_tai_san_row.empty:
        raise ValueError("Không tìm thấy chỉ tiêu 'TỔNG CỘNG TÀI SẢN'.")

    tong_tai_san_N_1 = tong_tai_san_row['Năm trước'].iloc[0]
    tong_tai_san_N = tong_tai_san_row['Năm sau'].iloc[0]

    # Xử lý giá trị 0 cho mẫu số
    divisor_N_1 = tong_tai_san_N_1 if tong_tai_san_N_1 != 0 else 1e-9
    divisor_N = tong_tai_san_N if tong_tai_san_N != 0 else 1e-9

    # Tính tỷ trọng với mẫu số đã được xử lý
    df['Tỷ trọng Năm trước (%)'] = (df['Năm trước'] / divisor_N_1) * 100
    df['Tỷ trọng Năm sau (%)'] = (df['Năm sau'] / divisor_N) * 100
    
    return df

# --- Hàm gọi API Gemini (Đã điều chỉnh cho chế độ Chat) ---
@st.cache_resource
def setup_gemini_client(api_key):
    """Thiết lập Client Gemini (Cache để tránh khởi tạo nhiều lần)"""
    return genai.Client(api_key=api_key)

def get_ai_response(data_for_ai, user_prompt, api_key):
    """Gửi dữ liệu phân tích và câu hỏi của người dùng đến Gemini API."""
    try:
        client = setup_gemini_client(api_key)
        model_name = 'gemini-2.5-flash' 

        # Tạo prompt tổng hợp
        # Thêm ngữ cảnh phân tích vào prompt của người dùng
        prompt = f"""
        Bạn là một chuyên gia phân tích tài chính chuyên nghiệp. Nhiệm vụ của bạn là trả lời các câu hỏi của người dùng một cách chính xác, chuyên nghiệp và DỰA TRÊN DỮ LIỆU TÀI CHÍNH đã được phân tích sau đây.

        **Dữ liệu Phân tích:**
        {data_for_ai}

        **Câu hỏi của người dùng:** {user_prompt}
        
        Hãy trả lời bằng Tiếng Việt.
        """

        response = client.models.generate_content(
            model=model_name,
            contents=prompt
        )
        return response.text

    except APIError as e:
        return f"Lỗi gọi Gemini API: Vui lòng kiểm tra Khóa API hoặc giới hạn sử dụng. Chi tiết lỗi: {e}"
    except Exception as e:
        return f"Đã xảy ra lỗi không xác định: {e}"


# --- Chức năng 1: Tải File ---
uploaded_file = st.file_uploader(
    "1. Tải file Excel Báo cáo Tài chính (Chỉ tiêu | Năm trước | Năm sau)",
    type=['xlsx', 'xls']
)

# Biến để lưu trữ dữ liệu đã xử lý dưới dạng markdown, dùng cho Chatbot
data_for_ai_markdown = None
chat_enabled = False

if uploaded_file is not None:
    try:
        df_raw = pd.read_excel(uploaded_file)
        
        # Tiền xử lý: Đảm bảo chỉ có 3 cột quan trọng
        df_raw.columns = ['Chỉ tiêu', 'Năm trước', 'Năm sau']
        
        # Xử lý dữ liệu
        df_processed = process_financial_data(df_raw.copy())
        chat_enabled = True # Bật chat sau khi xử lý thành công

        if df_processed is not None:
            
            # --- Chức năng 2 & 3: Hiển thị Kết quả (Giữ nguyên) ---
            st.subheader("2. Tốc độ Tăng trưởng & 3. Tỷ trọng Cơ cấu Tài sản")
            st.dataframe(df_processed.style.format({
                'Năm trước': '{:,.0f}',
                'Năm sau': '{:,.0f}',
                'Tốc độ tăng trưởng (%)': '{:.2f}%',
                'Tỷ trọng Năm trước (%)': '{:.2f}%',
                'Tỷ trọng Năm sau (%)': '{:.2f}%'
            }), use_container_width=True)
            
            # --- Chức năng 4: Tính Chỉ số Tài chính (Giữ nguyên) ---
            st.subheader("4. Các Chỉ số Tài chính Cơ bản")
            
            # Khởi tạo giá trị mặc định
            thanh_toan_hien_hanh_N = "N/A"
            thanh_toan_hien_hanh_N_1 = "N/A"

            try:
                # Lấy Tài sản ngắn hạn
                tsnh_n = df_processed[df_processed['Chỉ tiêu'].str.contains('TÀI SẢN NGẮN HẠN', case=False, na=False)]['Năm sau'].iloc[0]
                tsnh_n_1 = df_processed[df_processed['Chỉ tiêu'].str.contains('TÀI SẢN NGẮN HẠN', case=False, na=False)]['Năm trước'].iloc[0]

                # Lấy Nợ ngắn hạn
                no_ngan_han_N = df_processed[df_processed['Chỉ tiêu'].str.contains('NỢ NGẮN HẠN', case=False, na=False)]['Năm sau'].iloc[0]  
                no_ngan_han_N_1 = df_processed[df_processed['Chỉ tiêu'].str.contains('NỢ NGẮN HẠN', case=False, na=False)]['Năm trước'].iloc[0]

                # Tính toán, xử lý chia cho 0 bằng cách gán "N/A"
                thanh_toan_hien_hanh_N = tsnh_n / no_ngan_han_N if no_ngan_han_N != 0 else "N/A"
                thanh_toan_hien_hanh_N_1 = tsnh_n_1 / no_ngan_han_N_1 if no_ngan_han_N_1 != 0 else "N/A"
                
                # Hiển thị Metrics
                col1, col2 = st.columns(2)
                with col1:
                    st.metric(
                        label="Chỉ số Thanh toán Hiện hành (Năm trước)",
                        value=f"{thanh_toan_hien_hanh_N_1:.2f} lần" if isinstance(thanh_toan_hien_hanh_N_1, float) else "N/A"
                    )
                with col2:
                    if isinstance(thanh_toan_hien_hanh_N, float) and isinstance(thanh_toan_hien_hanh_N_1, float):
                         st.metric(
                            label="Chỉ số Thanh toán Hiện hành (Năm sau)",
                            value=f"{thanh_toan_hien_hanh_N:.2f} lần",
                            delta=f"{thanh_toan_hien_hanh_N - thanh_toan_hien_hanh_N_1:.2f}"
                        )
                    else:
                        st.metric(
                            label="Chỉ số Thanh toán Hiện hành (Năm sau)",
                            value="N/A"
                        )
                        
            except IndexError:
                st.warning("Thiếu chỉ tiêu 'TÀI SẢN NGẮN HẠN' hoặc 'NỢ NGẮN HẠN' để tính chỉ số.")
            except Exception as e:
                 st.warning(f"Lỗi tính toán chỉ số tài chính: {e}")
                
            # Chuẩn bị dữ liệu để gửi cho AI (Context cho Chatbot)
            data_for_ai_markdown = pd.DataFrame({
                'Chỉ tiêu': [
                    'Toàn bộ Bảng phân tích (dữ liệu thô)', 
                    'Tăng trưởng Tài sản ngắn hạn (%)', 
                    'Thanh toán hiện hành (N-1)', 
                    'Thanh toán hiện hành (N)'
                ],
                'Giá trị': [
                    df_processed.to_markdown(index=False),
                    f"{df_processed[df_processed['Chỉ tiêu'].str.contains('TÀI SẢN NGẮN HẠN', case=False, na=False)]['Tốc độ tăng trưởng (%)'].iloc[0]:.2f}%" if "TÀI SẢN NGẮN HẠN" in df_processed['Chỉ tiêu'].str.upper().values else "N/A",
                    f"{thanh_toan_hien_hanh_N_1:.2f}" if isinstance(thanh_toan_hien_hanh_N_1, float) else "N/A",
                    f"{thanh_toan_hien_hanh_N:.2f}" if isinstance(thanh_toan_hien_hanh_N, float) else "N/A"
                ]
            }).to_markdown(index=False)
            
            # Cập nhật tin nhắn chào sau khi dữ liệu đã tải
            if st.session_state["chat_messages"][0]["content"].startswith("Chào bạn! Vui lòng tải lên"):
                 st.session_state["chat_messages"] = [
                    {"role": "assistant", "content": "Tuyệt vời! Dữ liệu đã sẵn sàng. Bây giờ bạn có thể hỏi tôi bất kỳ điều gì về tình hình tài chính của doanh nghiệp này, ví dụ: 'Nhận xét về tốc độ tăng trưởng tài sản' hoặc 'Khả năng thanh toán hiện hành có thay đổi không?'"}
                ]
            
            # ----------------------------------------------------------------------------------
            # --- CHỨC NĂNG 5: CHATBOT TƯƠNG TÁC VỚI GEMINI (Thay thế hoàn toàn nút bấm) ---
            # ----------------------------------------------------------------------------------
            st.subheader("5. Chatbot Phân tích Tài chính AI 💬")

            # 1. Hiển thị lịch sử chat
            for message in st.session_state["chat_messages"]:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])

            # 2. Xử lý đầu vào từ người dùng
            prompt = st.chat_input("Hỏi AI về tình hình tài chính hoặc tăng trưởng...")
            
            if prompt:
                api_key = st.secrets.get("GEMINI_API_KEY") 
                
                if not api_key:
                    # Thêm tin nhắn lỗi vào lịch sử chat
                    error_message = "Lỗi: Không tìm thấy Khóa API. Vui lòng cấu hình Khóa 'GEMINI_API_KEY' trong Streamlit Secrets."
                    st.error(error_message)
                    st.session_state["chat_messages"].append({"role": "user", "content": prompt})
                    st.session_state["chat_messages"].append({"role": "assistant", "content": error_message})
                else:
                    # Thêm tin nhắn người dùng vào lịch sử và hiển thị
                    st.session_state["chat_messages"].append({"role": "user", "content": prompt})
                    with st.chat_message("user"):
                        st.markdown(prompt)
                    
                    # Hiển thị tin nhắn chờ từ AI và gọi API
                    with st.chat_message("assistant"):
                        with st.spinner('Đang gửi dữ liệu và chờ Gemini phân tích...'):
                            
                            # Gọi hàm AI với prompt mới
                            # Sử dụng hàm mới get_ai_response
                            ai_result = get_ai_response(data_for_ai_markdown, prompt, api_key)
                            
                            # Hiển thị kết quả của AI
                            st.markdown(ai_result)
                            
                            # Thêm phản hồi của AI vào lịch sử
                            st.session_state["chat_messages"].append({"role": "assistant", "content": ai_result})
                            
                    # Tự động cuộn trang xuống để thấy tin nhắn mới nhất
                    st.experimental_rerun() # Dùng rerun để cập nhật UI, mặc dù thường không cần thiết

    except ValueError as ve:
        st.error(f"Lỗi cấu trúc dữ liệu: {ve}")
    except Exception as e:
        st.error(f"Có lỗi xảy ra khi đọc hoặc xử lý file: {e}. Vui lòng kiểm tra định dạng file.")

else:
    # Hiển thị lịch sử chat ban đầu (khi chưa tải file)
    for message in st.session_state["chat_messages"]:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])
    
    # Khung chat input bị vô hiệu hóa nếu chưa có file
    st.chat_input("Vui lòng tải file để bắt đầu hỏi đáp...", disabled=True)
