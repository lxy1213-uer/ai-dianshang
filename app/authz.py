"""坐席 RBAC：角色、权限矩阵与身份解析。

设计原则：
- demo 模式（默认）：所有请求视为管理员，行为与加 RBAC 之前完全一致，课程演示不中断；
- strict 模式（AGENT_RBAC_ENABLED=1）：只信 X-Staff-Token 里的签名坐席身份，
  忽略请求体里的 assignee / operator / X-Agent-Name，杜绝伪造坐席身份。

用户身份（Authorization: Bearer <user token>）与坐席身份（X-Staff-Token）互不干扰：
前者决定"这个用户能看哪些会话"，后者决定"这个坐席能做什么操作"。
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum

from .auth import verify_staff_token


class Role(str, Enum):
    VIEWER = "viewer"        # 只读：看队列、看漏斗
    AGENT = "agent"          # 坐席：认领、回复、查工单、回写状态
    SUPERVISOR = "supervisor"  # 主管：归档
    ADMIN = "admin"          # 管理员：切模型、签发 token


ROLE_RANK = {Role.VIEWER: 0, Role.AGENT: 1, Role.SUPERVISOR: 2, Role.ADMIN: 3}


class Permission(str, Enum):
    DESK_READ = "desk:read"
    DESK_CLAIM = "desk:claim"
    DESK_REPLY = "desk:reply"
    DESK_RESOLVE = "desk:resolve"
    DESK_REASSIGN = "desk:reassign"
    TICKET_READ = "ticket:read"
    TICKET_STATUS = "ticket:status"
    REFUND_EXECUTE = "refund:execute"      # 资金类操作，最小角色 supervisor
    MODEL_SWITCH = "model:switch"
    METRICS_READ = "metrics:read"
    AUDIT_READ = "audit:read"  # 决策审计驾驶舱：跨会话只读，含资金/护栏事件


# 每个权限所需的最低角色
PERMISSION_MIN_ROLE = {
    Permission.DESK_READ: Role.VIEWER,
    Permission.METRICS_READ: Role.VIEWER,
    Permission.AUDIT_READ: Role.SUPERVISOR,
    Permission.TICKET_READ: Role.AGENT,
    Permission.DESK_CLAIM: Role.AGENT,
    Permission.DESK_REPLY: Role.AGENT,
    Permission.TICKET_STATUS: Role.AGENT,
    Permission.REFUND_EXECUTE: Role.SUPERVISOR,
    Permission.DESK_RESOLVE: Role.SUPERVISOR,
    Permission.DESK_REASSIGN: Role.SUPERVISOR,
    Permission.MODEL_SWITCH: Role.ADMIN,
}


@dataclass(frozen=True)
class StaffIdentity:
    name: str
    role: Role

    @property
    def role_value(self) -> str:
        return self.role.value


class AuthzError(Exception):
    """鉴权失败：status 决定 401（未认证）还是 403（无权限）。"""

    def __init__(self, status: int, code: str, message: str):
        super().__init__(message)
        self.status = status
        self.code = code
        self.message = message


def parse_role(value: str | None) -> Role | None:
    try:
        return Role((value or "").strip().lower())
    except ValueError:
        return None


def parse_accounts(raw: str) -> dict[str, dict[str, str]]:
    """解析 AGENT_STAFF_ACCOUNTS：name:password:role[:user_id];name2:...

    非法条目直接忽略（fail closed，不会因此放行）。密码明文只用于演示比对，
    生产环境应替换为 SSO 或密码哈希。
    """
    accounts: dict[str, dict[str, str]] = {}
    for chunk in (raw or "").split(";"):
        chunk = chunk.strip()
        if not chunk:
            continue
        parts = [p.strip() for p in chunk.split(":")]
        if len(parts) < 3:
            continue
        name, password, role_raw = parts[0], parts[1], parts[2]
        role = parse_role(role_raw)
        if not name or not password or not role:
            continue
        accounts[name] = {
            "password": password,
            "role": role.value,
            "user_id": parts[3] if len(parts) > 3 and parts[3] else name,
        }
    return accounts


def resolve_staff(request, settings) -> StaffIdentity | None:
    """从请求解析坐席身份。demo 模式直接给管理员，strict 模式只信签名 token。"""
    if not settings.rbac_enabled:
        return StaffIdentity(name="demo", role=Role.ADMIN)

    header = request.headers.get("x-staff-token", "")
    data = verify_staff_token(header.strip(), settings.auth_secret)
    if not data:
        return None
    role = parse_role(data.get("role"))
    name = str(data.get("sub") or "").strip()
    if not role or not name:
        return None
    return StaffIdentity(name=name, role=role)


def require_permission(request, settings, permission: Permission) -> StaffIdentity:
    """校验权限，失败抛 AuthzError（由 main.py 映射成 401/403）。"""
    staff = resolve_staff(request, settings)
    if staff is None:
        raise AuthzError(401, "unauthenticated", "缺少有效的坐席身份，请先登录")
    required = PERMISSION_MIN_ROLE[permission]
    if ROLE_RANK[staff.role] < ROLE_RANK[required]:
        raise AuthzError(
            403, "forbidden", f"当前角色 {staff.role_value} 无权执行 {permission.value}，需 {required.value} 或以上"
        )
    return staff
