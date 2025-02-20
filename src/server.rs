use actix_web::{web, App, HttpServer, HttpResponse, Responder};
use serde::{Serialize, Deserialize};
use std::path::PathBuf;
use std::sync::Arc;
use tokenizers::Tokenizer;

use crate::model::{Llama, SessionManager};
use log::info;

#[derive(Serialize, Deserialize)]
struct ChatRequest {
    session_id: String,
    prompt: String,
}

#[derive(Serialize)]
struct ChatResponse {
    response: String,
    session_id: String,
}

#[derive(Serialize, Deserialize)]
struct SessionCreateResponse {
    session_id: String,
}

#[derive(Serialize)]
struct HistoryResponse {
    history: Vec<String>,
}

#[derive(Serialize)]
struct ClearResponse {
    success: bool,
}

#[derive(Serialize, Deserialize)]
struct RollbackRequest {
    session_id: String,
    version_index: usize,
}

#[derive(Serialize)]
struct RollbackResponse {
    success: bool,
    history: Vec<String>,
}

// 添加新的结构体用于会话列表响应
#[derive(Serialize)]
struct SessionListResponse {
    sessions: Vec<String>,
}

/// 应用状态，包含共享模型、Tokenizer 和多会话管理器
struct AppState {
    model: Arc<Llama<f32>>,
    tokenizer: Tokenizer,
    session_manager: Arc<SessionManager>,
}

/// 创建新会话，返回 session_id
async fn create_session(data: web::Data<AppState>) -> impl Responder {
    let session_id = data.session_manager.create_session(data.model.clone());
    HttpResponse::Ok().json(SessionCreateResponse { session_id })
}

/// 聊天接口
async fn chat_endpoint(req: web::Json<ChatRequest>, data: web::Data<AppState>) -> impl Responder {
    let session_id = &req.session_id;
    let prompt = &req.prompt;
    // 从 SessionManager 中获取会话（Arc<Mutex<ChatSession>>）
    if let Some(session_arc) = data.session_manager.get_session(session_id) {
        // 锁定 session 后再调用相关方法
        let mut session = session_arc.lock().unwrap();
        session.add_user_message(prompt.clone());
        let full_prompt = session.build_prompt();
        info!("Full prompt: {}", full_prompt);
        
        let encoding = data.tokenizer.encode(full_prompt.clone(), true)
            .unwrap_or_else(|err| panic!("tokenizer encode error: {:?}", err));
        
        let prompt_ids = encoding.get_ids().to_vec();
        
        let output_ids = session.chat(&prompt_ids, 100, 0.8, 30, 1.0);
        
        let response_text = data.tokenizer.decode(&output_ids, true)
            .unwrap_or_else(|err| panic!("tokenizer decode error: {:?}", err));
        
        session.add_assistant_message(response_text.clone());
        HttpResponse::Ok().json(ChatResponse {
            response: response_text,
            session_id: session_id.clone(),
        })
    } else {
        HttpResponse::BadRequest().body("Invalid session_id")
    }
}

/// 获取历史会话记录接口，通过查询参数传递 session_id
async fn history_endpoint(query: web::Query<ChatRequest>, data: web::Data<AppState>) -> impl Responder {
    if let Some(session_arc) = data.session_manager.get_session(&query.session_id) {
        let session = session_arc.lock().unwrap();
        let history = session.get_history();
        HttpResponse::Ok().json(HistoryResponse { history })
    } else {
        HttpResponse::BadRequest().body("Invalid session_id")
    }
}

/// 清理当前会话接口
#[derive(Deserialize)]
struct SessionRequest {
    session_id: String,
}
async fn clear_endpoint(req: web::Json<SessionRequest>, data: web::Data<AppState>) -> impl Responder {
    if let Some(session_arc) = data.session_manager.get_session(&req.session_id) {
        let mut session = session_arc.lock().unwrap();
        session.clear();
        HttpResponse::Ok().json(ClearResponse { success: true })
    } else {
        HttpResponse::BadRequest().body("Invalid session_id")
    }
}

/// 回滚当前会话接口
async fn rollback_endpoint(req: web::Json<RollbackRequest>, data: web::Data<AppState>) -> impl Responder {
    if let Some(session_arc) = data.session_manager.get_session(&req.session_id) {
        let mut session = session_arc.lock().unwrap();
        session.rollback(req.version_index);
        let history = session.get_history();
        HttpResponse::Ok().json(RollbackResponse {
            success: true,
            history,
        })
    } else {
        HttpResponse::BadRequest().body("Invalid session_id")
    }
}

/// 获取会话列表的处理函数
async fn list_sessions(data: web::Data<AppState>) -> impl Responder {
    let sessions = data.session_manager.list_sessions();
    HttpResponse::Ok().json(SessionListResponse { sessions })
}

/// 首页，返回包含单个当前会话以及会话列表切换功能的 HTML 页面
async fn index() -> impl Responder {
    HttpResponse::Ok().body(
r#"
<html>
  <head>
    <meta charset="UTF-8">
    <title>多会话聊天系统</title>
    <style>
      /* 整体采用左右分栏布局 */
      body {
        font-family: Arial, sans-serif;
        display: flex;
        margin: 0;
        padding: 0;
        height: 100vh;
      }
      /* 左侧会话列表 */
      #session-list {
        width: 220px;
        border-right: 1px solid #ccc;
        padding: 10px;
        box-sizing: border-box;
        background-color: #f7f7f7;
      }
      #session-list h2 {
        font-size: 18px;
        margin: 0 0 10px 0;
      }
      #session-list ul {
        list-style: none;
        padding: 0;
        margin: 0;
      }
      #session-list ul li {
        padding: 8px;
        cursor: pointer;
        border-bottom: 1px solid #ddd;
      }
      #session-list ul li.active {
        background-color: #eee;
      }
      /* 右侧聊天区 */
      #chat-container {
        flex-grow: 1;
        padding: 10px;
        box-sizing: border-box;
        display: flex;
        flex-direction: column;
      }
      #chat-container h2 {
        margin: 0 0 10px 0;
      }
      #chat-history {
        flex-grow: 1;
        overflow-y: auto;
        border: 1px solid #ddd;
        margin-bottom: 10px;
        padding: 10px;
        background-color: #fff;
      }
      #chat-controls {
        display: flex;
      }
      #chat-input {
        flex-grow: 1;
        padding: 8px;
        font-size: 16px;
      }
      #send-btn, #clear-btn {
        padding: 8px 12px;
        margin-left: 5px;
        font-size: 16px;
      }
    </style>
  </head>
  <body>
    <div id="session-list">
      <h2>会话列表</h2>
      <button id="new-session">新建会话</button>
      <ul id="sessions"></ul>
    </div>
    <div id="chat-container">
      <h2>当前会话: <span id="active-session-id">无</span></h2>
      <button id="clear-btn">清空屏幕</button>
      <div id="chat-history"></div>
      <div id="chat-controls">
        <input type="text" id="chat-input" placeholder="输入消息...">
        <button id="send-btn">发送</button>
      </div>
    </div>
    <script>
      let activeSessionId = null;
      let sessions = {};
      const sessionsList = document.getElementById("sessions");
      const activeSessionSpan = document.getElementById("active-session-id");
      const chatHistory = document.getElementById("chat-history");
      const chatInput = document.getElementById("chat-input");
      
      // 页面加载时获取现有会话列表
      function loadExistingSessions() {
          fetch("/session/list")
              .then(response => response.json())
              .then(data => {
                  // 清空现有会话列表
                  sessionsList.innerHTML = "";
                  sessions = {};
                  
                  // 添加所有会话到列表
                  data.sessions.forEach(sessionId => {
                      sessions[sessionId] = [];  // 初始化会话历史记录
                      addSessionToList(sessionId);
                  });
                  
                  // 如果有会话，激活第一个
                  if (data.sessions.length > 0) {
                      setActiveSession(data.sessions[0]);
                  }
              });
      }
      
      // 页面加载时立即执行
      loadExistingSessions();
      
      // 新建会话按钮：调用 /session/create 接口，创建会话后加入列表并设为活动
      document.getElementById("new-session").addEventListener("click", function() {
          fetch("/session/create")
            .then(response => response.json())
            .then(data => {
              let sessionId = data.session_id;
              sessions[sessionId] = [];  // 初始化新会话的历史记录
              addSessionToList(sessionId);
              setActiveSession(sessionId);
            });
      });
      
      // 添加会话项到列表
      function addSessionToList(sessionId) {
          const li = document.createElement("li");
          li.textContent = sessionId;
          li.dataset.sessionId = sessionId;
          li.addEventListener("click", function() {
              setActiveSession(sessionId);
          });
          sessionsList.appendChild(li);
      }
      
      // 设置当前活动会话，并加载历史记录
      function setActiveSession(sessionId) {
          // 保存当前会话的聊天记录到 sessions 对象
          if (activeSessionId) {
              sessions[activeSessionId] = Array.from(chatHistory.children).map(p => p.textContent);
          }
          
          activeSessionId = sessionId;
          activeSessionSpan.textContent = sessionId;
          
          // 更新会话列表项的激活状态
          const items = sessionsList.querySelectorAll("li");
          items.forEach(item => {
              item.classList.remove("active");
              if (item.dataset.sessionId === sessionId) {
                  item.classList.add("active");
              }
          });
          
          // 显示选中会话的历史记录
          chatHistory.innerHTML = "";
          if (sessions[sessionId]) {
              sessions[sessionId].forEach(message => {
                  const p = document.createElement("p");
                  p.textContent = message;
                  chatHistory.appendChild(p);
              });
          } else {
              // 如果没有缓存的历史记录，从服务器加载
              loadHistory();
          }
      }
      
      // 加载当前会话历史记录
      function loadHistory() {
          if (!activeSessionId) return;
          fetch("/history?session_id=" + activeSessionId)
            .then(response => response.json())
            .then(data => {
               sessions[activeSessionId] = data.history;  // 缓存历史记录
               chatHistory.innerHTML = "";
               data.history.forEach(message => {
                  const p = document.createElement("p");
                  p.textContent = message;
                  chatHistory.appendChild(p);
               });
            });
      }
      
      // 发送消息：调用 /chat 接口，并将回复显示在历史记录中
      document.getElementById("send-btn").addEventListener("click", function(){
          const message = chatInput.value;
          if(message.trim() === "" || !activeSessionId) return;
          const p = document.createElement("p");
          p.textContent = "User: " + message;
          chatHistory.appendChild(p);
          
          fetch("/chat", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ session_id: activeSessionId, prompt: message })
          })
          .then(resp => resp.json())
          .then(res => {
              const p2 = document.createElement("p");
              p2.textContent = "Assistant: " + res.response;
              chatHistory.appendChild(p2);
          });
          
          chatInput.value = "";
      });
      
      // 清空当前会话屏幕：调用 /clear 接口后清空聊天记录
      document.getElementById("clear-btn").addEventListener("click", function(){
          if(!activeSessionId) return;
          fetch("/clear", {
             method: "POST",
             headers: { "Content-Type": "application/json" },
             body: JSON.stringify({ session_id: activeSessionId })
          })
          .then(resp => resp.json())
          .then(res => {
             chatHistory.innerHTML = "";
          });
      });
    </script>
  </body>
</html>
"#
    )
}

/// 启动 Web 服务
#[actix_web::main]
pub async fn main() -> std::io::Result<()> {
    // 设置模型目录，根据实际情况修改
    let project_dir = env!("CARGO_MANIFEST_DIR");
    let model_dir = PathBuf::from(project_dir).join("models").join("chat");
    
    // 加载模型和 Tokenizer
    let llama = Llama::from_safetensors(&model_dir);
    let tokenizer_path = model_dir.join("tokenizer.json");
    let tokenizer = Tokenizer::from_file(tokenizer_path)
        .unwrap_or_else(|err| panic!("failed to load tokenizer: {:?}", err));
    
    // 使用 Arc 包装模型
    let shared_model = Arc::new(llama);
    // 初始化 SessionManager，并包装在 Arc 中
    let session_manager = Arc::new(SessionManager::new());
    
    let data = web::Data::new(AppState {
        model: shared_model,
        tokenizer,
        session_manager, // 已经是 Arc<SessionManager>
    });
    
    println!("启动服务器：127.0.0.1:8080");
    HttpServer::new(move || {
        App::new()
            .app_data(data.clone())
            .route("/", web::get().to(index))
            .route("/session/create", web::get().to(create_session))
            .route("/session/list", web::get().to(list_sessions))  // 新增路由
            .route("/chat", web::post().to(chat_endpoint))
            .route("/history", web::get().to(history_endpoint))
            .route("/clear", web::post().to(clear_endpoint))
            .route("/rollback", web::post().to(rollback_endpoint))
    })
    .bind("127.0.0.1:8080")?
    .run()
    .await
}
