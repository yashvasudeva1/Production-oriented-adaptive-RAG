from __future__ import annotations

from email import policy
from email.parser import BytesParser
from pathlib import Path
from typing import Iterable

from ..base import BaseLoader
from ..exceptions import OptionalDependencyError
from ..models import DocumentBlock


class EMLLoader(BaseLoader):
    name = "eml-email-stdlib"
    extensions = (".eml",)

    def load_blocks(self, path: Path) -> Iterable[DocumentBlock]:
        with path.open("rb") as f:
            message = BytesParser(policy=policy.default).parse(f)

        headers = []
        for key in ("From", "To", "Cc", "Date", "Subject"):
            value = message.get(key)
            if value:
                headers.append(f"{key}: {value}")

        if headers:
            yield DocumentBlock(
                block_id="",
                text="\n".join(headers),
                block_type="email_headers",
                source_locator="email:headers",
            )

        body = _extract_email_body(message)
        if body:
            yield DocumentBlock(
                block_id="",
                text=body,
                block_type="email_body",
                source_locator="email:body",
            )


class MSGLoader(BaseLoader):
    name = "msg-extract-msg"
    extensions = (".msg",)

    def load_blocks(self, path: Path) -> Iterable[DocumentBlock]:
        try:
            import extract_msg
        except ImportError as exc:
            raise OptionalDependencyError(
                "MSG ingestion requires `extract-msg`."
            ) from exc

        message = extract_msg.Message(str(path))
        headers = []
        for label, value in (
            ("From", message.sender),
            ("To", message.to),
            ("Cc", message.cc),
            ("Subject", message.subject),
            ("Date", message.date),
        ):
            if value:
                headers.append(f"{label}: {value}")

        if headers:
            yield DocumentBlock(
                block_id="",
                text="\n".join(headers),
                block_type="email_headers",
                source_locator="msg:headers",
            )

        if message.body:
            yield DocumentBlock(
                block_id="",
                text=message.body,
                block_type="email_body",
                source_locator="msg:body",
            )


def _extract_email_body(message) -> str:
    if message.is_multipart():
        plain_parts = []
        html_parts = []

        for part in message.walk():
            content_type = part.get_content_type()
            if content_type == "text/plain":
                plain_parts.append(part.get_content())
            elif content_type == "text/html":
                html_parts.append(part.get_content())

        if plain_parts:
            return "\n\n".join(plain_parts)
        if html_parts:
            try:
                from bs4 import BeautifulSoup
                return BeautifulSoup(
                    "\n".join(html_parts),
                    "html.parser"
                ).get_text("\n", strip=True)
            except ImportError:
                return "\n".join(html_parts)

    try:
        return message.get_content()
    except Exception:
        return ""
