"""Web 服务：聊天接口、会话查询、健康检查（第 2 周 + 第 8 周）。"""

from __future__ import annotations

from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException, Query, Request
from urllib.parse import quote

from fastapi.responses import FileResponse, JSONResponse, Response
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, field_validator

from .agent.coordinator import Coordinator
from .agent.copilot import CopilotError
from .agent.diagnosis import DiagnosisError
from .agent.flow import TransitionError
from .auth import issue_staff_token, issue_token, resolve_user
from .authz import AuthzError, Permission, parse_accounts, require_permission
from .config import APP_DIR, Settings
from .domain.order_sources import OrderSourceError
from .domain.tickets import TicketError
from .identity import build_identity_provider
from .ratelimit import build_limiter, enforce
from .refund import RefundError
from .sla_scheduler import build_scheduler

STATIC_DIR = APP_DIR / "static"


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = Settings.load()
    app.state.settings = settings
    app.state.coordinator = Coordinator(settings)
    # 三个可插拔后端默认走零配置沙箱：进程内限流 / 账号表登录 / 拉队列触发 SLA。
    # 配置 Redis、SSO IdP、后台调度即可替换，聊天与状态机代码不用改。
    app.state.rate_limiter = build_limiter(settings)
    app.state.identity_provider = build_identity_provider(settings)
    app.state.sla_scheduler = build_scheduler(settings)
    from .settlement import RefundCallbackVerifier
    app.state.refund_callback_verifier = RefundCallbackVerifier(
        settings.refund_callback_secret, settings.refund_callback_tolerance
    )
    app.state.sla_scheduler.start(lambda: app.state.coordinator.desk.check_sla())
    try:
        yield
    finally:
        app.state.sla_scheduler.stop()
        app.state.coordinator.close()


app = FastAPI(
    title="电商 AI 智能客服与售后自动化系统",
    description="AGI Nexus 课程项目：查订单、答政策、判资格、建工单，全流程可演示。",
    version="1.0.0",
    lifespan=lifespan,
)

if STATIC_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")


class ChatRequest(BaseModel):
    session_id: str = Field(min_length=1, max_length=64, description="浏览器生成的会话 ID")
    # OIDC 的 sub 可能超过 32 字符（如 UUID），放宽到 64 避免真实 IdP 登录后被请求体拦住。
    user_id: str = Field(default="u_1001", max_length=64, description="当前登录用户（demo 模式）")
    message: str = Field(min_length=1, max_length=500, description="用户消息")

    @field_validator("message")
    @classmethod
    def message_not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("消息不能为空白")
        return value.strip()

    @field_validator("session_id", "user_id")
    @classmethod
    def strip_text(cls, value: str) -> str:
        return value.strip()


class DiagnosisRequest(BaseModel):
    question: str = Field(min_length=1, max_length=500, description="用户的售后问题原话")
    user_id: str = Field(default="u_1001", max_length=64, description="当前登录用户（demo 模式）")
    order_id: str | None = Field(default=None, max_length=64, description="可选订单号；不填时按商品描述尝试唯一匹配")

    @field_validator("question", "order_id")
    @classmethod
    def strip_text(cls, value: str | None) -> str | None:
        return value.strip() if isinstance(value, str) else value


class DiagnosisExecuteRequest(BaseModel):
    decision_id: str = Field(min_length=1, max_length=64, description="诊断接口下发的决策 ID")
    user_id: str = Field(default="u_1001", max_length=64)
    action: str = Field(default="create_ticket", max_length=32, description="create_ticket 或 handoff")
    confirm_text: str = Field(min_length=1, max_length=64, description="精确确认语，防止误点执行")

    @field_validator("decision_id", "action", "confirm_text")
    @classmethod
    def strip_text(cls, value: str) -> str:
        return value.strip()


class TokenRequest(BaseModel):
    user_id: str = Field(default="u_1001", max_length=64)


class LoginRequest(BaseModel):
    username: str = Field(min_length=1, max_length=64)
    # OIDC id_token 是 JWT，通常 600~1200 字符；128 会在 Pydantic 层直接 422。
    password: str = Field(min_length=1, max_length=4096)


class DeskClaimRequest(BaseModel):
    assignee: str = Field(min_length=1, max_length=64, description="坐席工号/姓名")


class DeskReplyRequest(BaseModel):
    assignee: str = Field(min_length=1, max_length=64)
    text: str = Field(min_length=1, max_length=1000)

    @field_validator("text")
    @classmethod
    def text_not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("回复内容不能为空白")
        return value.strip()


class DeskReassignRequest(BaseModel):
    assignee: str = Field(min_length=1, max_length=64, description="目标坐席工号/姓名")

    @field_validator("assignee")
    @classmethod
    def assignee_not_blank(cls, value: str) -> str:
        if not value.strip():
            raise ValueError("目标坐席不能为空白")
        return value.strip()


# 坐席快捷回复模板：不来自模型、不承载业务事实，只是话术片段（坐席点选后可再编辑）。
QUICK_REPLIES = (
    {"id": "greeting", "label": "开场致意", "text": "您好，我是售后专员，已看到您的诉求，马上为您处理。"},
    {"id": "accepted", "label": "已受理", "text": "您的申请已受理，我们会尽快核实并更新进度。"},
    {"id": "need_evidence", "label": "需补凭证", "text": "麻烦您补充问题照片或视频，便于我们核实，谢谢配合。"},
    {"id": "refund_timing", "label": "退款时效", "text": "质检通过后 1-3 个工作日内发起退款，到账时间以支付渠道为准。"},
    {"id": "checking", "label": "正在核实", "text": "我正在为您核实订单信息，请稍等片刻。"},
    {"id": "closing", "label": "结束致谢", "text": "问题已为您处理完毕，感谢您的耐心等待，如有疑问随时联系我们。"},
)


class TicketStatusRequest(BaseModel):
    status: str = Field(min_length=1, max_length=32, description="目标状态，来自 /api/ticket-statuses")
    operator: str = Field(default="agent", max_length=64, description="操作者标识")
    idempotency_key: str | None = Field(default=None, max_length=128, description="回写幂等键")

    @field_validator("status", "operator")
    @classmethod
    def strip_text(cls, value: str) -> str:
        return value.strip()


class RefundRequest(BaseModel):
    """退款请求：幂等键由服务端固定为 refund:<ticket_id>，不接受调用方传入。"""
    operator: str = Field(default="agent", max_length=64)
    amount: float | None = Field(default=None, ge=0, description="演示金额，不发生真实资金流转")
    currency: str = Field(default="CNY", max_length=8)


class ModelSwitchRequest(BaseModel):
    role: str = Field(description="main（表达层）或 classify（理解层）")
    model_id: str = Field(min_length=1, max_length=200, description="模型名，来自 /api/models")

    @field_validator("role")
    @classmethod
    def role_valid(cls, value: str) -> str:
        value = value.strip()
        if value not in ("main", "classify"):
            raise ValueError("role 必须是 main 或 classify")
        return value


def settings_of(request: Request) -> Settings:
    return request.app.state.settings


def coordinator(request: Request) -> Coordinator:
    instance = getattr(request.app.state, "coordinator", None)
    if instance is None:  # 极端情况下 lifespan 未执行（例如直接调用函数测试）
        raise HTTPException(status_code=503, detail="服务尚未就绪")
    return instance


def identity(request: Request, claimed: str | None) -> str:
    """解析请求身份：signed 模式只信签名 token；demo 模式要求显式 user_id。"""
    uid = resolve_user(request, claimed, settings_of(request))
    if not uid:
        raise HTTPException(
            status_code=401,
            detail="缺少身份标识：请提供 user_id，或在 signed 模式下携带有效 Authorization token",
        )
    return uid


def assert_owner(request: Request, session_id: str, user_id: str) -> None:
    """所有查询类接口都必须校验会话归属，避免凭 session_id 跨用户读取。"""
    session = coordinator(request).db.get_session(session_id)
    if session and session["user_id"] != user_id:
        raise HTTPException(status_code=403, detail="无权访问该会话的数据")


@app.exception_handler(AuthzError)
async def authz_error_handler(request: Request, exc: AuthzError):
    return JSONResponse(
        status_code=exc.status,
        content={"detail": exc.message, "code": exc.code},
    )


@app.exception_handler(TransitionError)
async def transition_error_handler(request: Request, exc: TransitionError):
    return JSONResponse(
        status_code=409,
        content={"detail": f"会话状态冲突：{exc}。请重试或结束本次会话后重新开始。"},
    )


from .domain.agent_desk import DeskError

_STATUS_ERROR_CODES = {
    "not_found": 404,
    "invalid_status": 400,
    "invalid_transition": 409,
    "version_conflict": 409,
    "already_claimed": 409,
    "not_claimed": 409,
    "resolved": 409,
    "forbidden": 403,
}


@app.exception_handler(TicketError)
async def ticket_error_handler(request: Request, exc: TicketError):
    return JSONResponse(
        status_code=_STATUS_ERROR_CODES.get(exc.code, 400),
        content={"detail": exc.message, "code": exc.code},
    )


@app.exception_handler(DeskError)
async def desk_error_handler(request: Request, exc: DeskError):
    return JSONResponse(
        status_code=_STATUS_ERROR_CODES.get(exc.code, 400),
        content={"detail": exc.message, "code": exc.code},
    )


@app.exception_handler(CopilotError)
async def copilot_error_handler(request: Request, exc: CopilotError):
    return JSONResponse(
        status_code=exc.status,
        content={"detail": {"code": exc.code, "message": exc.message}},
    )


@app.get("/", include_in_schema=False)
def index():
    page = STATIC_DIR / "index.html"
    if not page.exists():
        raise HTTPException(status_code=404, detail="前端页面缺失")
    # 前端改动需要即时生效，禁止浏览器缓存旧版页面（否则模型切换器点了没反应）
    return FileResponse(
        page,
        headers={"Cache-Control": "no-store, no-cache, must-revalidate, max-age=0"},
    )


@app.get("/desk", include_in_schema=False)
def desk_page():
    page = STATIC_DIR / "desk.html"
    if not page.exists():
        raise HTTPException(status_code=404, detail="坐席工作台页面缺失")
    return FileResponse(page, headers={"Cache-Control": "no-store, no-cache, must-revalidate, max-age=0"})


@app.get("/metrics", include_in_schema=False)
def metrics_page():
    page = STATIC_DIR / "metrics.html"
    if not page.exists():
        raise HTTPException(status_code=404, detail="埋点漏斗页面缺失")
    return FileResponse(page, headers={"Cache-Control": "no-store, no-cache, must-revalidate, max-age=0"})


@app.get("/portal", include_in_schema=False)
def portal_page():
    page = STATIC_DIR / "portal.html"
    if not page.exists():
        raise HTTPException(status_code=404, detail="自助进度页面缺失")
    return FileResponse(page, headers={"Cache-Control": "no-store, no-cache, must-revalidate, max-age=0"})


@app.get("/diagnose", include_in_schema=False)
def diagnose_page():
    page = STATIC_DIR / "diagnosis.html"
    if not page.exists():
        raise HTTPException(status_code=404, detail="售后诊断页面缺失")
    return FileResponse(page, headers={"Cache-Control": "no-store, no-cache, must-revalidate, max-age=0"})


@app.get("/audit", include_in_schema=False)
def audit_page():
    page = STATIC_DIR / "audit.html"
    if not page.exists():
        raise HTTPException(status_code=404, detail="决策审计驾驶舱页面缺失")
    return FileResponse(page, headers={"Cache-Control": "no-store, no-cache, must-revalidate, max-age=0"})


@app.get("/api/health", summary="健康检查（部署验收用）")
def health(request: Request):
    return coordinator(request).health()


@app.get("/api/models", summary="可切换模型列表与当前选择")
def list_models(request: Request):
    """返回脱敏后的模型选项；api_key 只在服务端内存，不进响应。"""
    return coordinator(request).models_view()


@app.post("/api/models/switch", summary="运行时切换某一层的模型")
def switch_model(payload: ModelSwitchRequest, request: Request):
    """切换立即生效于下一条消息，不需要重启服务。strict 模式要求管理员角色。"""
    enforce(request, "switch", settings_of(request).rate_limit_switch)
    require_permission(request, settings_of(request), Permission.MODEL_SWITCH)
    try:
        return coordinator(request).switch_model(payload.role, payload.model_id)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))


@app.post("/api/auth/token", summary="演示用身份签发")
def auth_token(payload: TokenRequest, request: Request):
    """signed 模式返回 HMAC 签名 token；demo 模式返回 token=null。

    真实系统里这个入口应由登录态 / SSO 保护，本项目只演示"服务端签发 + 归属校验"。
    """
    enforce(request, "auth", settings_of(request).rate_limit_auth)
    settings = settings_of(request)
    user_id = payload.user_id.strip() or "u_1001"
    if not settings.signed_auth_enabled:
        return {"auth_mode": "demo", "user_id": user_id, "token": None}
    return {
        "auth_mode": "signed",
        "user_id": user_id,
        "token": issue_token(user_id, settings.auth_secret, settings.auth_ttl_seconds),
    }


@app.post("/api/auth/login", summary="坐席登录（校验账号表，签发用户 token + 坐席 token）")
def auth_login(payload: LoginRequest, request: Request):
    """strict 模式校验 AGENT_STAFF_ACCOUNTS 里的账号密码，签发两类 token；
    demo 模式返回免鉴权身份，保持课程演示零配置可用。"""
    enforce(request, "auth", settings_of(request).rate_limit_auth)
    settings = settings_of(request)
    if not settings.rbac_enabled:
        return {
            "auth_mode": "demo",
            "user_id": payload.username or "u_1001",
            "token": None,
            "staff_token": None,
            "role": "admin",
        }
    identity = request.app.state.identity_provider.authenticate(
        payload.username.strip(), payload.password
    )
    if identity is None:
        raise HTTPException(status_code=401, detail="坐席账号或密码错误")
    secret = settings.auth_secret
    if not secret:
        raise HTTPException(status_code=503, detail="未配置 AGENT_AUTH_SECRET，无法签发坐席 token")
    return {
        "auth_mode": "strict",
        "user_id": identity.user_id,
        "token": issue_token(identity.user_id, secret, settings.auth_ttl_seconds),
        "staff_token": issue_staff_token(
            identity.username, identity.role, secret, settings.staff_ttl_seconds
        ),
        "role": identity.role,
    }


@app.post("/api/chat", summary="与客服 Agent 对话")
def chat(payload: ChatRequest, request: Request):
    enforce(request, "chat", settings_of(request).rate_limit_chat)
    user_id = identity(request, payload.user_id)
    result = coordinator(request).chat(payload.session_id, user_id, payload.message)
    return result.to_dict()


@app.post("/api/aftersale/diagnose", summary="售后诊断：事实、条款证据、缺口与可执行动作")
def aftersale_diagnose(payload: DiagnosisRequest, request: Request):
    enforce(request, "chat", settings_of(request).rate_limit_chat)
    uid = identity(request, payload.user_id)
    try:
        return coordinator(request).diagnosis.diagnose(
            payload.question, uid, payload.order_id
        )
    except DiagnosisError as exc:
        raise HTTPException(status_code=exc.status, detail={"code": exc.code, "message": exc.message})


@app.post("/api/aftersale/execute", summary="执行诊断动作（服务端复核 + 显式确认）")
def aftersale_execute(payload: DiagnosisExecuteRequest, request: Request):
    enforce(request, "chat", settings_of(request).rate_limit_chat)
    uid = identity(request, payload.user_id)
    try:
        return coordinator(request).diagnosis.execute(
            payload.decision_id, uid, payload.action, payload.confirm_text
        )
    except DiagnosisError as exc:
        raise HTTPException(status_code=exc.status, detail={"code": exc.code, "message": exc.message})


@app.get("/api/orders", summary="列出当前登录用户自己的订单（按订单源配置）")
def list_orders(
    request: Request,
    user_id: str | None = Query(default=None),
    limit: int = Query(default=50, ge=1, le=50),
    status: str | None = Query(default=None, max_length=32),
):
    """按服务端身份隔离；user_id 只用于 demo 模式，signed 模式只信 token。"""
    uid = identity(request, user_id)
    normalized_status = (status or "").strip().lower()
    allowed = {"", "paid", "shipped", "delivered", "refunded", "closed"}
    if normalized_status not in allowed:
        raise HTTPException(status_code=400, detail="status 必须是 paid/shipped/delivered/refunded/closed")
    c = coordinator(request)
    try:
        orders = c.list_my_orders(uid, limit=limit, status=normalized_status or None)
    except OrderSourceError as exc:
        raise HTTPException(
            status_code=503,
            detail={"code": "order_source_unavailable", "message": str(exc)},
        ) from exc
    return {
        "user_id": uid,
        "source": c.order_source.describe(),
        "orders": orders,
    }


@app.get("/api/sessions/{session_id}", summary="查询会话状态（刷新后继续办理）")
def get_session(session_id: str, request: Request, user_id: str | None = Query(default=None)):
    uid = identity(request, user_id)
    assert_owner(request, session_id, uid)
    return coordinator(request).session_view(session_id, uid)


@app.post("/api/sessions/{session_id}/reset", summary="重置会话")
def reset_session(session_id: str, request: Request, user_id: str | None = Query(default=None)):
    uid = identity(request, user_id)
    assert_owner(request, session_id, uid)
    coordinator(request).flow.reset(session_id, uid)
    return {"ok": True, "session_id": session_id, "state": "IDLE"}


@app.get("/api/ticket-statuses", summary="工单状态枚举与迁移表（坐席/前端共用）")
def ticket_statuses():
    from .domain.status import ALLOWED, NEXT_STEP, TERMINAL, TicketStatus
    return {
        "statuses": [
            {
                "id": s.value, "label": s.label,
                "next_step": NEXT_STEP[s],
                "terminal": s in TERMINAL,
                "transitions": sorted(t.value for t in ALLOWED[s]),
            }
            for s in TicketStatus
        ]
    }


@app.get("/api/tickets", summary="查询售后工单（需校验会话归属）")
def list_tickets(request: Request, session_id: str = Query(...), user_id: str | None = Query(default=None)):
    uid = identity(request, user_id)
    assert_owner(request, session_id, uid)
    return {"tickets": coordinator(request).db.list_tickets(session_id, user_id=uid)}


@app.get("/api/portal/tickets", summary="自助进度查询：按当前用户列出全部工单（不问也能查）")
def portal_tickets(request: Request, user_id: str | None = Query(default=None)):
    # 只按身份归属过滤，不依赖 session_id——用户不必和客服对话也能查自己的工单。
    # strict 模式下 identity() 只信签名 token，天然隔离；demo 模式需显式 user_id。
    uid = identity(request, user_id)
    tickets = coordinator(request).db.list_tickets(user_id=uid, limit=50)
    return {
        "user_id": uid,
        "tickets": [
            {
                "ticket_id": t["ticket_id"],
                "order_id": t["order_id"],
                "reason": t["reason"],
                "status": t["status"],
                "created_at": t["created_at"],
                "updated_at": t.get("updated_at") or t["created_at"],
            }
            for t in tickets
        ],
    }


@app.get("/api/tickets/{ticket_id}", summary="工单详情（状态/时间线/下一步）")
def get_ticket(ticket_id: str, request: Request, user_id: str | None = Query(default=None)):
    require_permission(request, settings_of(request), Permission.TICKET_READ)
    uid = identity(request, user_id)
    ticket = coordinator(request).db.get_ticket(ticket_id)
    if not ticket:
        raise HTTPException(status_code=404, detail="工单不存在")
    # 归属校验：按工单所属 user_id 隔离，避免凭工单号跨用户读取
    if ticket["user_id"] != uid:
        raise HTTPException(status_code=403, detail="无权访问该工单")
    return coordinator(request).tickets.detail(ticket_id)


@app.post("/api/tickets/{ticket_id}/status", summary="回写工单状态（含幂等与并发冲突）")
def update_ticket_status(ticket_id: str, payload: TicketStatusRequest, request: Request):
    # strict 模式下角色来自签名 token，操作者名字也以 token 为准，忽略请求体伪造的 operator；
    # demo 模式沿用X-Agent-Name / operator，保持现有演示行为。
    staff = require_permission(request, settings_of(request), Permission.TICKET_STATUS)
    operator = staff.name if settings_of(request).rbac_enabled else (
        request.headers.get("X-Agent-Name") or payload.operator or "system"
    )
    return coordinator(request).tickets.update_status(
        ticket_id,
        payload.status,
        operator=operator,
        idempotency_key=payload.idempotency_key,
    )


@app.post("/api/tickets/{ticket_id}/refund", summary="执行退款（沙箱默认，绝不重复打款）")
def execute_refund(ticket_id: str, payload: RefundRequest, request: Request):
    """资金类操作：strict 模式需 supervisor 及以上，操作者以签名 token 为准。"""
    s = settings_of(request)
    staff = require_permission(request, s, Permission.REFUND_EXECUTE)
    operator = staff.name if s.rbac_enabled else (
        request.headers.get("X-Agent-Name") or payload.operator or "system"
    )
    try:
        return coordinator(request).refunds.execute(
            ticket_id, operator=operator, amount=payload.amount, currency=payload.currency,
        )
    except RefundError as exc:
        raise HTTPException(status_code=409, detail={"code": exc.code, "message": exc.message})


@app.get("/api/tickets/{ticket_id}/refund", summary="查询退款流水（幂等状态）")
def refund_status(ticket_id: str, request: Request):
    require_permission(request, settings_of(request), Permission.TICKET_READ)
    return {
        "ticket_id": ticket_id,
        "refund": coordinator(request).refunds.status(ticket_id),
    }


@app.post("/api/refund/callback", summary="支付网关退款回调（验签通过只记录，不自动打款/改状态）")
async def refund_callback(request: Request):
    """网关异步通知。验签失败 401；验签通过也**只落事件交人工核对**，
    绝不据此自动推进工单状态或再次调用退款网关——资金类变更必须人工确认。"""
    from .settlement import CallbackError, RefundCallbackVerifier

    verifier: RefundCallbackVerifier = request.app.state.refund_callback_verifier
    raw = await request.body()
    try:
        payload = verifier.verify(
            raw,
            request.headers.get(RefundCallbackVerifier.HEADER_SIGNATURE, ""),
            request.headers.get(RefundCallbackVerifier.HEADER_TIMESTAMP, ""),
        )
    except CallbackError as exc:
        raise HTTPException(status_code=401, detail=str(exc))
    c = coordinator(request)
    c.db.log_event(
        "", "", "refund_callback_received",
        {"refund_id": payload.get("refund_id"), "ticket_id": payload.get("ticket_id"),
         "status": payload.get("status"), "verified": True},
    )
    return {"accepted": True, "requires_manual_review": True}


@app.post("/api/refund/reconcile", summary="对账：网关对账文件与本地流水比对（只输出差异）")
async def refund_reconcile(request: Request):
    """上传网关对账 CSV/JSON，返回差异清单与建议动作；不修改任何状态。"""
    require_permission(request, settings_of(request), Permission.REFUND_EXECUTE)
    from .settlement import differences_to_dict, parse_ledger, reconcile

    body = await request.body()
    if not body:
        raise HTTPException(status_code=400, detail="对账文件为空")
    try:
        fmt = request.query_params.get("format", "auto")
        gateway = parse_ledger(body.decode("utf-8"), fmt)
    except (ValueError, UnicodeDecodeError) as exc:
        raise HTTPException(status_code=400, detail=f"对账文件解析失败：{exc}")
    c = coordinator(request)
    local = []
    for row in c.db.all_refunds():
        local.append(row)
    diffs = reconcile(local, gateway)
    return {"gateway_rows": len(gateway), "local_rows": len(local),
            "differences": differences_to_dict(diffs)}


@app.get("/api/trace/{session_id}", summary="查询最近的事件与轨迹（需校验会话归属）")
def trace(session_id: str, request: Request, user_id: str | None = Query(default=None)):
    uid = identity(request, user_id)
    assert_owner(request, session_id, uid)
    return {"events": coordinator(request).db.list_events(session_id, limit=30)}


@app.get("/api/usage", summary="查询会话 Token 用量与计费口径（需校验会话归属）")
def usage(request: Request, session_id: str = Query(...), user_id: str | None = Query(default=None)):
    """实测 token 只来自 API usage；规则模式只有字符估算，费用为 0。

    金额是按可配置单价换算的估算费用，不是供应商账单。
    """
    uid = identity(request, user_id)
    assert_owner(request, session_id, uid)
    return coordinator(request).usage_view(session_id)


# ---------- 事项 B：人工坐席工作台 ----------
@app.get("/api/desk/sla-status", summary="SLA 超时通知配置状态（供坐席台展示）")
def desk_sla_status(request: Request):
    require_permission(request, settings_of(request), Permission.DESK_READ)
    c = coordinator(request)
    return {
        "notifier_configured": c.notifier.configured,
        "alert_minutes": settings_of(request).sla_alert_minutes,
        "last_notified": c.desk.last_notified,
    }


@app.post("/api/desk/sla-check", summary="手动触发 SLA 超时扫描与通知（演示/定时任务用）")
def desk_sla_check(request: Request):
    require_permission(request, settings_of(request), Permission.DESK_READ)
    records = coordinator(request).desk.check_sla()
    return {"new_overdue": len(records), "queue_ids": [r["queue_id"] for r in records]}


@app.get("/api/desk/queue", summary="坐席队列（支持筛选）")
def desk_queue(
    request: Request,
    state: str | None = Query(default=None),
    assignee: str | None = Query(default=None),
    only_overdue: bool = Query(default=False),
):
    enforce(request, "desk", 0)
    require_permission(request, settings_of(request), Permission.DESK_READ)
    return {"items": coordinator(request).desk.list_queue(
        state=state, assignee=assignee, only_overdue=only_overdue
    )}


@app.get("/api/desk/queue/{queue_id}", summary="转人工记录详情")
def desk_detail(queue_id: str, request: Request):
    require_permission(request, settings_of(request), Permission.DESK_READ)
    return coordinator(request).desk.detail(queue_id)


@app.get("/api/desk/queue/{queue_id}/copilot", summary="坐席 Copilot 建议（只读，需坐席权限）")
def desk_copilot(queue_id: str, request: Request):
    require_permission(request, settings_of(request), Permission.DESK_READ)
    return coordinator(request).copilot.suggest(queue_id)


@app.post("/api/desk/queue/{queue_id}/claim", summary="认领会话（互斥）")
def desk_claim(queue_id: str, payload: DeskClaimRequest, request: Request):
    staff = require_permission(request, settings_of(request), Permission.DESK_CLAIM)
    assignee = staff.name if settings_of(request).rbac_enabled else payload.assignee.strip()
    return coordinator(request).desk.claim(queue_id, assignee)


@app.post("/api/desk/queue/{queue_id}/reply", summary="坐席回复（用户在原会话可见）")
def desk_reply(queue_id: str, payload: DeskReplyRequest, request: Request):
    staff = require_permission(request, settings_of(request), Permission.DESK_REPLY)
    assignee = staff.name if settings_of(request).rbac_enabled else payload.assignee.strip()
    return coordinator(request).desk.reply(queue_id, assignee, payload.text)


@app.post("/api/desk/queue/{queue_id}/resolve", summary="结束归档")
def desk_resolve(queue_id: str, request: Request):
    require_permission(request, settings_of(request), Permission.DESK_RESOLVE)
    return coordinator(request).desk.resolve(queue_id)


@app.post("/api/desk/queue/{queue_id}/reassign", summary="转派会话给另一名坐席（需主管权限）")
def desk_reassign(queue_id: str, payload: DeskReassignRequest, request: Request):
    staff = require_permission(request, settings_of(request), Permission.DESK_REASSIGN)
    return coordinator(request).desk.reassign(
        queue_id, payload.assignee, operator=staff.name if settings_of(request).rbac_enabled else None
    )


@app.get("/api/desk/quick-replies", summary="坐席快捷回复模板与可转派坐席名单")
def desk_quick_replies(request: Request):
    require_permission(request, settings_of(request), Permission.DESK_READ)
    s = settings_of(request)
    # strict 模式列出配置的坐席；demo 模式给几个演示工号，便于体验转派
    accounts = parse_accounts(s.staff_accounts)
    assignees = sorted(accounts) if accounts else ["坐席-02", "坐席-03", "主管-01"]
    return {"items": list(QUICK_REPLIES), "assignees": assignees}


# ---------- 事项 C：埋点漏斗 ----------
@app.get("/api/metrics/funnel", summary="转化漏斗（按时间筛选）")
def metrics_funnel(
    request: Request,
    date_from: str | None = Query(default=None, alias="from"),
    date_to: str | None = Query(default=None, alias="to"),
):
    require_permission(request, settings_of(request), Permission.METRICS_READ)
    return coordinator(request).funnel.funnel(date_from, date_to)


@app.get("/api/metrics/by-mode", summary="按单/多 Agent 分组的效果对比")
def metrics_by_mode(request: Request):
    require_permission(request, settings_of(request), Permission.METRICS_READ)
    return coordinator(request).funnel.by_mode()


@app.get("/api/metrics/event-dict", summary="统一事件字典")
def metrics_event_dict():
    from .domain.events import FUNNEL_LABELS, FUNNEL_STEPS, EventName
    return {
        "funnel_steps": [
            {"event": k, "label": FUNNEL_LABELS[k]} for k in FUNNEL_STEPS
        ],
        "all_events": [e.value for e in EventName],
    }


# ---------- 决策审计驾驶舱（跨会话只读，最低主管权限） ----------
@app.get("/api/audit/overview", summary="决策审计总览：决策量、执行率、资金动作、护栏拦截")
def audit_overview(
    request: Request,
    date_from: str | None = Query(default=None, alias="from"),
    date_to: str | None = Query(default=None, alias="to"),
):
    require_permission(request, settings_of(request), Permission.AUDIT_READ)
    return coordinator(request).audit.overview(date_from, date_to)


@app.get("/api/audit/decisions", summary="决策列表（按 verdict / 是否已执行 / 用户 / 时间筛选）")
def audit_decisions(
    request: Request,
    verdict: str | None = Query(default=None),
    executed: str | None = Query(default=None, pattern="^(yes|no)$"),
    user_id: str | None = Query(default=None),
    date_from: str | None = Query(default=None, alias="from"),
    date_to: str | None = Query(default=None, alias="to"),
    limit: int = Query(default=50, ge=1, le=200),
):
    require_permission(request, settings_of(request), Permission.AUDIT_READ)
    return coordinator(request).audit.decisions(
        verdict, executed, user_id, date_from, date_to, limit
    )


@app.get("/api/audit/decisions/{decision_id}", summary="决策详情：诊断 → 执行 → 工单 → 退款 → 护栏完整链")
def audit_decision_detail(decision_id: str, request: Request):
    require_permission(request, settings_of(request), Permission.AUDIT_READ)
    return coordinator(request).audit.detail(decision_id)


@app.get("/api/audit/events", summary="跨会话审计事件流（按类型 / 分组 / 时间过滤）")
def audit_events(
    request: Request,
    kinds: str | None = Query(default=None, description="逗号分隔的事件类型"),
    group: str | None = Query(default=None),
    date_from: str | None = Query(default=None, alias="from"),
    date_to: str | None = Query(default=None, alias="to"),
    limit: int = Query(default=100, ge=1, le=500),
):
    require_permission(request, settings_of(request), Permission.AUDIT_READ)
    selected = [k.strip() for k in kinds.split(",") if k.strip()] if kinds else None
    return coordinator(request).audit.events(selected, group, date_from, date_to, limit)


@app.get("/api/audit/risks", summary="风险清单：仍需人工关注的护栏拦截 / 资金异常 / SLA")
def audit_risks(request: Request, limit: int = Query(default=100, ge=1, le=500)):
    require_permission(request, settings_of(request), Permission.AUDIT_READ)
    return coordinator(request).audit.risks(limit)


# ---------- 诊断报告导出（只读：HTML 可打印为 PDF / xlsx 零依赖） ----------
def _render_export(report: dict, fmt: str, filename: str) -> Response:
    """统一渲染与响应头：HTML 内联可打印，xlsx 作为附件下载。

    HTTP 头是 latin-1，中文文件名必须用 RFC 5987 的 filename* 编码。
    """
    from .domain.export import ReportExporter

    allowed = set("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789-_")
    ascii_name = "report-" + "".join(c for c in filename if c in allowed)[:48]
    encoded = quote(filename, safe="")
    if fmt == "html":
        return Response(
            content=ReportExporter.render_html(report),
            media_type="text/html; charset=utf-8",
            headers={"Content-Disposition": f"inline; filename={ascii_name}.html; filename*=UTF-8''{encoded}.html"},
        )
    if fmt == "xlsx":
        return Response(
            content=ReportExporter.render_xlsx(report),
            media_type="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            headers={
                "Content-Disposition": f"attachment; filename={ascii_name}.xlsx; filename*=UTF-8''{encoded}.xlsx",
                "Cache-Control": "no-store, no-cache, must-revalidate, max-age=0",
            },
        )
    raise HTTPException(status_code=400, detail="format 必须是 html 或 xlsx")


@app.get("/api/reports/diagnosis/{decision_id}", summary="导出自己的诊断报告（用户视角，只读）")
def export_diagnosis_report(
    decision_id: str,
    request: Request,
    user_id: str | None = Query(default=None),
    fmt: str = Query(default="html", alias="format", pattern="^(html|xlsx)$"),
):
    """按服务端身份隔离；用户可查阅/导出过期诊断，但不能拿旧结论执行。"""
    uid = identity(request, user_id)
    c = coordinator(request)
    try:
        report = c.exports.build_diagnosis(decision_id, uid)
    except Exception as exc:
        status = getattr(exc, "status", 400)
        raise HTTPException(status_code=status, detail=getattr(exc, "message", str(exc)))
    return _render_export(report, fmt, f"诊断报告-{decision_id}")


@app.get("/api/reports/audit/{decision_id}", summary="导出完整决策审计报告（主管视角，只读）")
def export_audit_report(
    decision_id: str,
    request: Request,
    fmt: str = Query(default="html", alias="format", pattern="^(html|xlsx)$"),
):
    require_permission(request, settings_of(request), Permission.AUDIT_READ)
    c = coordinator(request)
    try:
        report = c.exports.build_audit(decision_id)
    except Exception as exc:
        status = getattr(exc, "status", 400)
        raise HTTPException(status_code=status, detail=getattr(exc, "message", str(exc)))
    return _render_export(report, fmt, f"决策审计报告-{decision_id}")
