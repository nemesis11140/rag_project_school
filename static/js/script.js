let conversations = [];
let currentConversationId = null;
let isSending = false;
let currentUser = null;
let authenticationCheck = null;
let pendingQuery = null;
let isAuthenticating = false;
const ACTIVE_CONVERSATION_SESSION_KEY = 'campus-rag-active-conversation';

document.addEventListener('DOMContentLoaded', function() {
    init();
});

async function init() {
    clearLegacyBrowserHistory();
    renderConversationList();
    renderWelcome();
    warmupModels();
    checkStatus();

    document.getElementById('query-input').addEventListener('keydown', function(e) {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            sendQuery();
        }
    });

    document.getElementById('auth-overlay').addEventListener('click', event => {
        if (event.target.id === 'auth-overlay') closeAuth();
    });
    document.addEventListener('keydown', event => {
        const overlay = document.getElementById('auth-overlay');
        if (overlay.classList.contains('hidden')) return;
        if (event.key === 'Escape') {
            event.preventDefault();
            closeAuth();
        } else if (event.key === 'Tab') {
            const controls = Array.from(overlay.querySelectorAll('button, input'))
                .filter(element => !element.disabled && element.getClientRects().length);
            const first = controls[0];
            const last = controls[controls.length - 1];
            if (event.shiftKey && document.activeElement === first) {
                event.preventDefault();
                last?.focus();
            } else if (!event.shiftKey && document.activeElement === last) {
                event.preventDefault();
                first?.focus();
            }
        }
    });

    authenticationCheck = checkAuthentication();
    await authenticationCheck;
}

function clearLegacyBrowserHistory() {
    // 旧版本曾把完整消息保存在 localStorage。升级后主动清理，后续不再写入。
    localStorage.removeItem('campus-rag-conversations-v1');
    localStorage.removeItem('campus-rag-active-conversation-v1');
}

async function requestJson(url, options = {}) {
    const response = await fetch(url, {
        credentials: 'same-origin',
        ...options,
        headers: {
            ...(options.body ? { 'Content-Type': 'application/json' } : {}),
            ...(options.headers || {})
        }
    });
    const payload = await response.json().catch(() => ({}));
    if (!response.ok) {
        const error = new Error(payload.detail || `请求失败（${response.status}）`);
        error.status = response.status;
        if (response.status === 401 && !url.startsWith('/api/auth/')) {
            enterGuestApp();
            showAuth();
        }
        throw error;
    }
    return payload;
}

// ---------- 登录与注册 ----------

async function checkAuthentication() {
    try {
        const payload = await requestJson('/api/auth/me');
        await enterAuthenticatedApp(payload.user);
    } catch (error) {
        enterGuestApp();
        if (error.status !== 401) {
            addSystemNotice('暂时无法确认登录状态，提问时可重新登录。');
        }
    }
}

function switchAuthMode(mode) {
    if (isAuthenticating) return;
    ['login', 'register', 'reset'].forEach(name => {
        document.getElementById(`${name}-form`).classList.toggle('hidden', mode !== name);
    });
    document.querySelectorAll('.auth-tab').forEach(tab => {
        tab.classList.toggle('active', tab.dataset.mode === mode);
    });
    document.getElementById('auth-title').textContent = mode === 'reset'
        ? '重置账户密码'
        : mode === 'register' ? '注册校园智能问答助手' : '登录校园智能问答助手';
    document.getElementById('auth-description').textContent = mode === 'reset'
        ? '输入已注册的用户名，设置新密码'
        : pendingQuery ? '登录后将继续发送刚才的问题，并保存对话记录'
        : '登录后即可提问，并保存你的对话记录';
    if (mode === 'reset') {
        document.getElementById('reset-username').value = document.getElementById('login-username').value.trim();
    }
    showAuthError('');
    showAuthSuccess('');
    document.getElementById(`${mode}-username`).focus();
}

function enterGuestApp() {
    currentUser = null;
    conversations = [];
    setActiveConversation(null);
    document.getElementById('current-user').textContent = '访客';
    document.getElementById('login-btn').classList.remove('hidden');
    document.getElementById('logout-btn').classList.add('hidden');
    renderConversationList();
    renderCurrentConversation();
}

function showAuth() {
    document.getElementById('auth-description').textContent = pendingQuery
        ? '登录后将继续发送刚才的问题，并保存对话记录'
        : '登录后即可提问，并保存你的对话记录';
    document.getElementById('auth-overlay').classList.remove('hidden');
    document.querySelector('.app-container').inert = true;
    switchAuthMode('login');
}

function closeAuth() {
    if (isAuthenticating) return;
    pendingQuery = null;
    document.getElementById('auth-overlay').classList.add('hidden');
    document.querySelector('.app-container').inert = false;
    document.getElementById('login-form').reset();
    document.getElementById('register-form').reset();
    document.getElementById('reset-form').reset();
    showAuthError('');
    showAuthSuccess('');
    document.getElementById('query-input').focus();
}

function showAuthError(message) {
    document.getElementById('auth-error').textContent = message || '';
    if (message) showAuthSuccess('');
}

function showAuthSuccess(message) {
    const notice = document.getElementById('auth-success');
    notice.textContent = message || '';
    notice.classList.toggle('hidden', !message);
}

function setAuthFormLoading(form, loading) {
    isAuthenticating = loading;
    document.getElementById('auth-close').disabled = loading;
    document.querySelectorAll('.auth-tab').forEach(tab => { tab.disabled = loading; });
    form.querySelectorAll('input, button').forEach(element => {
        element.disabled = loading;
    });
}

async function enterAuthenticatedApp(user) {
    currentUser = user;
    document.getElementById('current-user').textContent = user.username;
    conversations = [];
    currentConversationId = null;
    await initializeConversations();
    if (!currentUser) return;
    document.getElementById('login-btn').classList.add('hidden');
    document.getElementById('logout-btn').classList.remove('hidden');
    document.getElementById('auth-overlay').classList.add('hidden');
    document.querySelector('.app-container').inert = false;
    document.getElementById('query-input').focus();
}

async function resumePendingQuery() {
    if (!currentUser || !pendingQuery) return;
    document.getElementById('query-input').value = pendingQuery;
    pendingQuery = null;
    await sendQuery();
}

async function submitLogin(event) {
    event.preventDefault();
    const form = event.currentTarget;
    const username = document.getElementById('login-username').value.trim();
    const password = document.getElementById('login-password').value;
    showAuthError('');
    setAuthFormLoading(form, true);

    try {
        const payload = await requestJson('/api/auth/login', {
            method: 'POST',
            body: JSON.stringify({ username, password })
        });
        form.reset();
        await enterAuthenticatedApp(payload.user);
        await resumePendingQuery();
    } catch (error) {
        showAuthError(error.message);
    } finally {
        setAuthFormLoading(form, false);
    }
}

async function submitRegister(event) {
    event.preventDefault();
    const form = event.currentTarget;
    const username = document.getElementById('register-username').value.trim();
    const password = document.getElementById('register-password').value;
    const confirmation = document.getElementById('register-confirm-password').value;
    showAuthError('');

    if (password !== confirmation) {
        showAuthError('两次输入的密码不一致');
        return;
    }

    setAuthFormLoading(form, true);
    try {
        const payload = await requestJson('/api/auth/register', {
            method: 'POST',
            body: JSON.stringify({ username, password })
        });
        form.reset();
        await enterAuthenticatedApp(payload.user);
        await resumePendingQuery();
    } catch (error) {
        showAuthError(error.message);
    } finally {
        setAuthFormLoading(form, false);
    }
}

async function logout() {
    if (isSending) return;
    try {
        await requestJson('/api/auth/logout', { method: 'POST' });
    } catch (error) {
        console.error('退出失败:', error);
        addSystemNotice(`退出失败：${error.message}`);
        return;
    }
    pendingQuery = null;
    document.getElementById('query-input').value = '';
    enterGuestApp();
}

async function submitPasswordReset(event) {
    event.preventDefault();
    if (isAuthenticating) return;
    const form = event.currentTarget;
    const username = document.getElementById('reset-username').value.trim();
    const password = document.getElementById('reset-password').value;
    const confirmation = document.getElementById('reset-confirm-password').value;
    showAuthError('');
    showAuthSuccess('');
    if (password !== confirmation) {
        showAuthError('两次输入的新密码不一致');
        return;
    }
    setAuthFormLoading(form, true);
    let result = null;
    try {
        result = await requestJson('/api/auth/reset-password', {
            method: 'POST',
            body: JSON.stringify({ username, new_password: password })
        });
        form.reset();
    } catch (error) {
        showAuthError(error.message);
    } finally {
        setAuthFormLoading(form, false);
    }
    if (result?.success) {
        switchAuthMode('login');
        document.getElementById('login-username').value = username;
        document.getElementById('login-password').value = '';
        showAuthSuccess(result.message);
        document.getElementById('login-password').focus();
    }
}

// ---------- MySQL 对话记录 ----------

async function initializeConversations() {
    try {
        await refreshConversationList();

        // sessionStorage 在刷新时保留、关闭标签页后清除，只记录当前会话 ID。
        // 会话列表与消息正文仍然全部从 MySQL 读取。
        const savedId = Number(sessionStorage.getItem(ACTIVE_CONVERSATION_SESSION_KEY));
        const savedConversation = Number.isInteger(savedId)
            ? conversations.find(item => Number(item.id) === savedId)
            : null;

        if (savedConversation) {
            setActiveConversation(savedConversation.id);
            await loadConversationMessages(savedConversation.id);
        } else {
            // 关闭页面后重新进入时，从一个空白的新对话开始。
            // 若数据库中已经有未使用的空会话则直接复用，避免产生空记录。
            const latestConversation = conversations[0];
            if (latestConversation && Number(latestConversation.message_count) === 0) {
                latestConversation.messages = [];
                setActiveConversation(latestConversation.id);
            } else {
                await createConversationOnServer();
            }
        }
        renderConversationList();
        renderCurrentConversation();
    } catch (error) {
        console.error('加载对话记录失败:', error);
        renderWelcome();
        addSystemNotice(`对话记录加载失败：${error.message}`);
    }
}

async function refreshConversationList() {
    const payload = await requestJson('/api/conversations');
    const oldMessages = new Map(
        conversations.map(item => [item.id, item.messages])
    );
    conversations = (payload.conversations || []).map(item => ({
        ...item,
        messages: oldMessages.has(item.id) ? oldMessages.get(item.id) : null
    }));
}

async function createConversationOnServer() {
    const payload = await requestJson('/api/conversations', {
        method: 'POST',
        body: JSON.stringify({ title: '新对话' })
    });
    const conversation = { ...payload.conversation, messages: [] };
    conversations.unshift(conversation);
    setActiveConversation(conversation.id);
    return conversation;
}

function setActiveConversation(id) {
    currentConversationId = id;
    if (id === null || id === undefined) {
        sessionStorage.removeItem(ACTIVE_CONVERSATION_SESSION_KEY);
    } else {
        sessionStorage.setItem(ACTIVE_CONVERSATION_SESSION_KEY, String(id));
    }
}

async function loadConversationMessages(id) {
    const payload = await requestJson(`/api/conversations/${id}/messages`);
    const conversation = conversations.find(item => item.id === id);
    if (conversation) {
        conversation.messages = payload.messages || [];
    }
    return conversation;
}

function getCurrentConversation() {
    return conversations.find(item => item.id === currentConversationId);
}

async function newConversation() {
    if (isSending) return;
    await authenticationCheck;
    if (!currentUser) {
        pendingQuery = null;
        document.getElementById('query-input').value = '';
        renderCurrentConversation();
        document.getElementById('query-input').focus();
        return;
    }

    const current = getCurrentConversation();
    if (current && Array.isArray(current.messages) && current.messages.length === 0) {
        document.getElementById('query-input').focus();
        return;
    }

    try {
        await createConversationOnServer();
        renderConversationList();
        renderCurrentConversation();
        document.getElementById('query-input').focus();
    } catch (error) {
        console.error('新建对话失败:', error);
        addSystemNotice(`新建对话失败：${error.message}`);
    }
}

async function selectConversation(id) {
    if (isSending || id === currentConversationId) return;

    const conversation = conversations.find(item => item.id === id);
    if (!conversation) return;

    try {
        if (!Array.isArray(conversation.messages)) {
            await loadConversationMessages(id);
        }
        setActiveConversation(id);
        renderConversationList();
        renderCurrentConversation();
    } catch (error) {
        console.error('读取对话失败:', error);
        addSystemNotice(`读取对话失败：${error.message}`);
    }
}

async function deleteConversation(event, id) {
    event.stopPropagation();
    if (isSending) return;

    try {
        await requestJson(`/api/conversations/${id}`, { method: 'DELETE' });
        conversations = conversations.filter(item => item.id !== id);

        if (!conversations.length) {
            await createConversationOnServer();
        } else if (currentConversationId === id) {
            setActiveConversation(conversations[0].id);
            if (!Array.isArray(conversations[0].messages)) {
                await loadConversationMessages(currentConversationId);
            }
        }

        renderConversationList();
        renderCurrentConversation();
    } catch (error) {
        console.error('删除对话失败:', error);
        addSystemNotice(`删除对话失败：${error.message}`);
    }
}

function renderConversationList() {
    const list = document.getElementById('conversation-list');
    list.innerHTML = '';

    if (!currentUser) {
        const hint = document.createElement('p');
        hint.className = 'guest-history';
        hint.textContent = '登录后，你的对话记录会显示在这里。';
        list.appendChild(hint);
        return;
    }

    conversations.forEach(conversation => {
        const item = document.createElement('button');
        item.type = 'button';
        item.className = `conversation-item${conversation.id === currentConversationId ? ' active' : ''}`;
        item.addEventListener('click', () => selectConversation(conversation.id));

        const title = document.createElement('span');
        title.className = 'conversation-title';
        title.textContent = conversation.title;

        const remove = document.createElement('span');
        remove.className = 'delete-conversation';
        remove.title = '删除对话';
        remove.textContent = '×';
        remove.addEventListener('click', event => deleteConversation(event, conversation.id));

        item.appendChild(title);
        item.appendChild(remove);
        list.appendChild(item);
    });
}

function renderCurrentConversation() {
    const container = document.getElementById('chat-messages');
    container.innerHTML = '';

    const conversation = getCurrentConversation();
    if (!conversation || !conversation.messages || conversation.messages.length === 0) {
        renderWelcome();
        return;
    }

    conversation.messages.forEach(message => addMessage(message.content, message.role));
    container.scrollTop = container.scrollHeight;
}

function renderWelcome() {
    const container = document.getElementById('chat-messages');
    if (!container) return;

    addMessage(
        '你好！我是小交，上海交通大学的智能校园问答助手。\n' +
        '我可以帮你查询学籍管理、奖学金、宿舍生活和办事流程等校园信息。',
        'assistant'
    );

    const panel = document.createElement('div');
    panel.className = 'suggestion-panel';

    const title = document.createElement('p');
    title.textContent = '你可以试着问：';
    panel.appendChild(title);

    const questions = document.createElement('div');
    questions.className = 'quick-questions';
    [
        ['📚', '图书馆借书期限是多久？'],
        ['🏆', '怎么申请奖学金？'],
        ['📋', '学生证丢了怎么补办？'],
        ['🍜', '食堂几点开门？'],
        ['🔄', '怎么申请转专业？'],
        ['⚠️', '在校打架会受到什么处分？']
    ].forEach(([icon, question]) => {
        const button = document.createElement('button');
        button.type = 'button';
        button.textContent = `${icon} ${question}`;
        button.addEventListener('click', () => askQuestion(question));
        questions.appendChild(button);
    });

    panel.appendChild(questions);
    container.appendChild(panel);
}

function addSystemNotice(text) {
    const container = document.getElementById('chat-messages');
    if (!container) return;
    const notice = document.createElement('p');
    notice.className = 'system-notice';
    notice.textContent = text;
    container.appendChild(notice);
}

// ---------- 服务状态 ----------

async function warmupModels() {
    try {
        await fetch('/api/warmup', { method: 'POST', credentials: 'same-origin' });
    } catch (error) {
        console.error('预热失败:', error);
    }
    checkStatus();
}

async function checkStatus() {
    try {
        const response = await fetch('/api/health', { credentials: 'same-origin' });
        const data = await response.json();
        const statusDot = document.querySelector('.status-dot');
        const statusText = document.querySelector('.status-text');

        if (data.models_ready) {
            statusDot.className = 'status-dot online';
            statusText.textContent = '系统就绪';
        } else if (data.warmup === 'failed') {
            statusDot.className = 'status-dot offline';
            statusText.textContent = data.problem?.message || data.error || '启动检查失败，请查看日志';
        } else if (data.system_ready) {
            statusDot.className = 'status-dot loading';
            statusText.textContent = '模型加载中，首个问题会稍慢…';
        } else {
            statusDot.className = 'status-dot offline';
            statusText.textContent = '系统未就绪';
        }
        statusText.title = data.problem?.hint || '';
    } catch (error) {
        console.error('健康检查失败:', error);
    }
}

// ---------- 消息收发 ----------

function askQuestion(question) {
    if (isSending) return;
    document.getElementById('query-input').value = question;
    sendQuery();
}

async function sendQuery() {
    const input = document.getElementById('query-input');
    const query = input.value.trim();
    if (!query || isSending) return;
    await authenticationCheck;
    if (isSending) return;
    if (!currentUser) {
        pendingQuery = query;
        showAuth();
        return;
    }

    const conversation = getCurrentConversation();
    if (!conversation) {
        addSystemNotice('对话尚未加载成功，请点击“新建对话”后重试。');
        return;
    }

    isSending = true;
    const sendBtn = document.getElementById('send-btn');
    sendBtn.disabled = true;
    input.disabled = true;

    const conversationId = conversation.id;
    if (!Array.isArray(conversation.messages)) conversation.messages = [];
    conversation.messages.push({ role: 'user', content: query });

    document.querySelector('.suggestion-panel')?.remove();
    addMessage(query, 'user');
    input.value = '';
    const typingMsg = addTypingIndicator();

    try {
        const result = await requestJson('/api/query', {
            method: 'POST',
            body: JSON.stringify({ query, conversation_id: conversationId })
        });

        conversation.messages.push({ role: 'assistant', content: result.answer });
        if (result.conversation) {
            Object.assign(conversation, result.conversation);
        }
        conversations = [
            conversation,
            ...conversations.filter(item => item.id !== conversationId)
        ];
        renderConversationList();
        if (currentConversationId === conversationId) {
            addMessage(result.answer, 'assistant');
        }
    } catch (error) {
        console.error('查询失败:', error);
        if (error.status === 401) {
            input.value = query;
            pendingQuery = query;
            showAuth();
            return;
        }
        try {
            await loadConversationMessages(conversationId);
            renderCurrentConversation();
        } catch (_) {
            // 保留原错误作为主要提示。
        }
        addSystemNotice(`发送失败：${error.message}`);
    } finally {
        typingMsg.stopThinking?.();
        typingMsg.remove();
        isSending = false;
        sendBtn.disabled = false;
        input.disabled = false;
        if (document.getElementById('auth-overlay').classList.contains('hidden')) {
            input.focus();
        }
    }
}

function addMessage(text, type) {
    const container = document.getElementById('chat-messages');
    const message = document.createElement('div');
    message.className = `message ${type}-message`;

    const avatar = document.createElement('div');
    avatar.className = 'message-avatar';
    avatar.textContent = type === 'assistant' ? '🤖' : '👤';

    const content = document.createElement('div');
    content.className = 'message-content';

    const textDiv = document.createElement('div');
    textDiv.className = 'message-text';
    textDiv.innerHTML = formatMessage(text);

    content.appendChild(textDiv);
    message.appendChild(avatar);
    message.appendChild(content);
    container.appendChild(message);
    container.scrollTop = container.scrollHeight;
    return message;
}

function addTypingIndicator() {
    const container = document.getElementById('chat-messages');
    const message = document.createElement('div');
    message.className = 'message assistant-message';

    const avatar = document.createElement('div');
    avatar.className = 'message-avatar';
    avatar.textContent = '🤖';

    const content = document.createElement('div');
    content.className = 'message-content';

    const indicator = document.createElement('div');
    indicator.className = 'typing-indicator';
    indicator.setAttribute('role', 'status');
    indicator.setAttribute('aria-live', 'polite');

    const label = document.createElement('span');
    label.className = 'thinking-label';
    label.textContent = '思考中';

    const dots = document.createElement('span');
    dots.className = 'thinking-dots';
    dots.innerHTML = '<span class="typing-dot"></span><span class="typing-dot"></span><span class="typing-dot"></span>';

    const elapsed = document.createElement('span');
    elapsed.className = 'thinking-elapsed';
    elapsed.textContent = '0 秒';

    indicator.appendChild(label);
    indicator.appendChild(dots);
    indicator.appendChild(elapsed);

    content.appendChild(indicator);
    message.appendChild(avatar);
    message.appendChild(content);
    container.appendChild(message);
    container.scrollTop = container.scrollHeight;

    let seconds = 0;
    const timer = window.setInterval(() => {
        seconds += 1;
        elapsed.textContent = `${seconds} 秒`;
    }, 1000);
    message.stopThinking = () => window.clearInterval(timer);
    return message;
}

function formatMessage(text) {
    if (!text) return '';
    return escapeHtml(String(text)).replace(/\n/g, '<br>');
}

function escapeHtml(text) {
    return text
        .replace(/&/g, '&amp;')
        .replace(/</g, '&lt;')
        .replace(/>/g, '&gt;')
        .replace(/"/g, '&quot;')
        .replace(/'/g, '&#039;');
}
