"""
beefriend —— 求加好友（反向好友申请）
bot 在群里 @ 某人，请对方把 bot 加为好友，并推送 bot 自己的名片。

流程：
1. 目标：@ 指定 / 指定 QQ / 从当前群随机或选最久未联系的人
2. 冷却：同一用户 cooldown_days 内只找一次
3. 话术：AI 结合（对方昵称 + 近 N 条群聊）生成自然话术；
   无 AI 时随机从 default_reasons 模板里挑，{name} 占位符替换昵称
4. 发送：群消息 = @对方 + 话术 + bot 个人名片，对方点名片就能加
5. 标记：发送成功后写冷却记录（复用 mark_befriend_sent）
"""

import asyncio
import random
import re

from aiocqhttp import CQHttp

from astrbot.api import logger
from astrbot.api.star import Context

from .config import PluginConfig
from .utils import get_ats


class BeFriendHandle:
    def __init__(self, context: Context, config: PluginConfig):
        self.context = context
        self.cfg = config
        self._task: asyncio.Task | None = None

    # ---------------- 客户端 ----------------

    def _client(self) -> CQHttp | None:
        """取 aiocqhttp 客户端（反向 WS），取不到返回 None"""
        try:
            insts = self.context.platform_manager.platform_insts
        except Exception:
            return None
        for inst in insts:
            get_client = getattr(inst, "get_client", None)
            if callable(get_client):
                try:
                    if client := get_client():
                        return client
                except Exception:
                    continue
        return None

    @staticmethod
    def _at_dict(uid: int, nickname: str = "") -> dict:
        """构造 @ 段：优先带真实昵称 name（协议端渲染成蓝色 @），
        拿不到昵称就只带 qq，由协议端自己解析群名片。
        绝不能让 name 显示成 QQ 号。"""
        data: dict = {"qq": str(uid)}
        if nickname and nickname != str(uid):
            data["name"] = nickname
        return {"type": "at", "data": data}

    # ---------------- 目标解析 ----------------

    async def _resolve_target(
        self,
        event,
        raw: str,
    ) -> tuple[int | None, str | None]:
        """
        解析目标，返回 (user_id, 共同群号或 None)
        优先级：@某人 > 数字QQ > 当前群随机/最久未联系
        """
        gid = event.get_group_id() if not event.is_private_chat() else ""
        client = self._client()
        if client is None:
            return None, None

        # 1. @ 指定
        if at_ids := get_ats(event, noself=True):
            uid = int(at_ids[0])
            return uid, gid or None

        # 2. 文本里给 QQ 号
        m = re.search(r"\d{5,12}", raw)
        if m:
            return int(m.group()), gid or None

        # 3. 兜底：从当前群成员里挑（私聊则无群，返回 None）
        if not gid:
            return None, None

        try:
            members = await client.get_group_member_list(group_id=int(gid))
        except Exception as e:
            logger.warning(f"求加好友获取群成员失败({gid}): {e}")
            return None, None

        self_id = str(event.get_self_id())
        candidates = [
            str(mb.get("user_id"))
            for mb in members
            if isinstance(mb, dict)
            and mb.get("user_id") is not None
            and str(mb.get("user_id")) != self_id
            and str(mb.get("user_id")) not in self.cfg.user_blacklist
            and not self.cfg.in_befriend_cooldown(mb.get("user_id"))
        ]
        if not candidates:
            return None, None

        if raw.strip():
            # 带了文字但没带 QQ/@：视为随机挑
            uid = str(random.choice(candidates))
        else:
            # 最久没找过的人优先
            uid = self.cfg.most_long_uncontacted(candidates) or random.choice(candidates)
        return int(uid), gid

    # ---------------- 话术 ----------------

    async def _make_reason(
        self,
        client: CQHttp,
        uid: int,
        group_id: str,
    ) -> tuple[str, str]:
        """生成求加好友的话术：AI 优先，失败降级模板。返回 (话术, 对方昵称)"""
        uid_str = str(uid)
        nickname = ""

        # 拿昵称（群名片 > 昵称），拿不到就留空（at 段不写 name，
        # 由协议端自己解析成群名片；千万别拿 QQ 号当 name，不然渲染成 @QQ号）
        if group_id and group_id.isdigit():
            try:
                info = await client.get_group_member_info(
                    group_id=int(group_id), user_id=uid
                )
                nickname = info.get("card") or info.get("nickname") or info.get("nick") or ""
            except Exception:
                pass
        if not nickname:
            try:
                info = await client.get_stranger_info(user_id=uid)
                nickname = info.get("nickname") or info.get("nick") or ""
            except Exception:
                pass
        if nickname == uid_str:
            nickname = ""

        # 最近聊天记录（仅群聊能取）
        history_text = ""
        if group_id and group_id.isdigit() and self.cfg.beefriend.history_count > 0:
            try:
                ret = await client.get_group_msg_history(
                    group_id=int(group_id),
                    count=int(self.cfg.beefriend.history_count),
                    reverseOrder=True,
                )
                msgs = ret.get("messages") if isinstance(ret, dict) else None
                if msgs:
                    lines = []
                    for m in msgs[: self.cfg.beefriend.history_count]:
                        sender = (m.get("sender") or {}).get("nickname", "?")
                        content = str(m.get("message", "")).replace("\n", " ")
                        lines.append(f"{sender}: {content}")
                    history_text = "\n".join(lines[-self.cfg.beefriend.history_count :])
            except Exception as e:
                logger.debug(f"求加好友获取聊天记录失败: {e}")

        reason = await self._ai_reason(nickname, history_text)
        if not reason:
            reason = self._template_reason(nickname)
        return reason, nickname

    async def _ai_reason(self, nickname: str, history_text: str) -> str:
        """用 LLM 生成一段自然的求加好友话术"""
        provider = None
        try:
            provider = await self.context.get_using_provider_async()
        except Exception:
            provider = None
        if provider is None:
            return ""

        try:
            sys_prompt = (
                "你在一个 QQ 群里，想加这个群里的人为 QQ 好友，请帮你想一段在群里 @ TA "
                "之后接的话术，请 TA 把你加为好友。要求：口语自然、简短（一两句话）、不油腻不尬、"
                "开头不要带对方昵称（@ 已经带了）、可以自然地提一下你和 TA 的共同点或 TA "
                "最近聊过的话题，末尾不用重复说明，因为后面会附你的名片。"
                "直接输出话术本身，不要任何解释、引号或前缀。"
            )
            user_prompt = f"对方昵称：{nickname}\n"
            if history_text:
                user_prompt += f"群里最近的聊天：\n{history_text}\n"
            user_prompt += "请生成求加好友的话术。"
            resp = await provider.text_chat(
                system_prompt=sys_prompt,
                prompt=user_prompt,
            )
            text = (resp.completion_text or "").strip()
            if text and len(text) <= 200:
                return text
        except Exception as e:
            logger.warning(f"AI 生成求加好友话术失败: {e}")
        return ""

    def _template_reason(self, nickname: str) -> str:
        """模板兜底：从 default_reasons 随机挑一条，替换 {name}"""
        reasons = self.cfg.beefriend.default_reasons or []
        if not reasons:
            return f"哈喽 {nickname}，我是群里的 bot，想加你为好友，方便的话把我加一下呗"
        tpl = random.choice(reasons)
        try:
            return tpl.format(name=nickname)
        except (KeyError, IndexError):
            return tpl.replace("{name}", nickname)

    # ---------------- 发送 ----------------

    async def _send_befriend(
        self,
        client: CQHttp,
        uid: int,
        reason: str,
        group_id: str,
        nickname: str,
    ) -> bool:
        """群消息：@对方（显示昵称）+ 话术 + bot 名片，对方点名片就能加好友"""
        if not group_id or not group_id.isdigit():
            return False

        payload = [
            self._at_dict(uid, nickname),
            {"type": "text", "data": {"text": f" {reason} "}},
        ]

        # 附 bot 自己的名片，方便对方直接点加
        try:
            login = await client.get_login_info()
            self_id = int(login.get("user_id"))
            payload.append(
                {"type": "contact", "data": {"type": "qq", "id": self_id}}
            )
        except Exception:
            pass  # 拿不到自己 id 就不发名片

        try:
            await client.send_group_msg(group_id=int(group_id), message=payload)
            return True
        except Exception as e:
            logger.error(f"求加好友群消息发送失败(uid={uid} gid={group_id}): {e}")
            return False

    # ---------------- 入口 ----------------

    async def befriend(self, event, raw: str) -> str:
        """手动触发入口：返回给用户的提示文本"""
        client = self._client()
        if client is None:
            return "未找到可用的 OneBot 客户端"

        uid, gid = await self._resolve_target(event, raw)
        if uid is None:
            return (
                "没有可找的人：可以 @ 一下群里的人，或直接给个 QQ 号；"
                "在群里直接发则从群成员里挑"
            )
        if not gid:
            return "需要在群里用哦，@ 不到群外的人"
        if self.cfg.in_befriend_cooldown(uid):
            left = self.cfg.befriend_cooldown_left(uid)
            hours = left / 3600
            return f"{uid} 还在冷却里，约 {hours:.1f} 小时后再找 TA"

        reason, nickname = await self._make_reason(client, uid, gid or "")
        ok = await self._send_befriend(client, uid, reason, gid or "", nickname)
        if not ok:
            return f"群消息发送失败了（可能不在 {gid} 群）"

        self.cfg.mark_befriend_sent(uid)
        return f"已在群里 @ {uid} 求加好友：{reason}"

    # ---------------- 后台任务 ----------------

    async def _run_befriend_once(self) -> None:
        """后台自动找一次"""
        if not self.cfg.beefriend.auto:
            return
        client = self._client()
        if client is None:
            logger.debug("beefriend: 客户端未就绪")
            return

        # 收集所有群成员候选，挑最久没找过的
        gid, uid = None, None
        try:
            groups = await client.get_group_list()
        except Exception as e:
            logger.warning(f"beefriend 获取群列表失败: {e}")
            return

        all_candidates: list[str] = []
        group_of_uid: dict[str, str] = {}
        for g in groups:
            gid_x = str(g.get("group_id"))
            if not gid_x.isdigit():
                continue
            try:
                members = await client.get_group_member_list(group_id=int(gid_x))
            except Exception:
                continue
            for mb in members:
                uid_x = str(mb.get("user_id"))
                if uid_x in self.cfg.user_blacklist:
                    continue
                if self.cfg.in_befriend_cooldown(uid_x):
                    continue
                if uid_x not in group_of_uid:
                    all_candidates.append(uid_x)
                    group_of_uid[uid_x] = gid_x

        if all_candidates:
            uid = self.cfg.most_long_uncontacted(all_candidates)
            gid = group_of_uid[uid] if uid else None

        if not uid:
            logger.debug("beefriend: 没有可找的人（全在冷却或无群成员）")
            return

        reason, nickname = await self._make_reason(client, int(uid), gid or "")
        ok = await self._send_befriend(
            client, int(uid), reason, gid or "", nickname
        )
        if ok:
            self.cfg.mark_befriend_sent(uid)
            logger.info(f"beefriend 自动求加好友成功: uid={uid} gid={gid}")
        await asyncio.sleep(self.cfg.beefriend.auto_delay)

    def start(self) -> None:
        """启动后台自动任务（间隔 auto_interval 分钟）"""
        if not self.cfg.beefriend.auto or self._task and not self._task.done():
            return
        self._task = asyncio.create_task(self._auto_loop())

    async def _auto_loop(self) -> None:
        while True:
            try:
                await asyncio.sleep(60)  # 启动后先等 60s 等平台就绪
                await self._run_befriend_once()
            except asyncio.CancelledError:
                return
            except Exception as e:
                logger.exception(f"beefriend 自动任务异常: {e}")
            await asyncio.sleep(self.cfg.beefriend.auto_interval * 60)

    def stop(self) -> None:
        if self._task and not self._task.done():
            self._task.cancel()
