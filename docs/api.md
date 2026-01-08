# Enterprise API

## 概要

| 項目 | 詳細 |
|------|------|
| 認証 | `X-API-Key: aicheckers_ent_xxx...` ヘッダー |
| レート制限 | なし |
| キー発行 | `/admin/enterprise/create-key` (管理者のみ) |
| 使用量確認 | `/admin/enterprise/usage-all` (管理者のみ) |
| データ保存 | `data/enterprise_keys.json`, `data/enterprise_usage.json` |

## 企業向けAPIキー発行手順

```bash
# 1. 管理者アカウントでサイトにログイン → JWTトークンを取得
# 2. 以下を実行（company_name, contact_emailを適宜変更）
curl -X POST https://api.aicheckers.net/admin/enterprise/create-key \
  -H "Authorization: Bearer YOUR_ADMIN_JWT" \
  -H "Content-Type: application/json" \
  -d '{"company_name": "株式会社Example", "contact_email": "api@example.co.jp", "plan": "standard", "expires_days": 365}'

# レスポンスにapi_keyが含まれる → これを企業に渡す
```

## 既存認証との共存

Enterprise APIは`X-API-Key`ヘッダーがある場合のみ有効。従来のJWT認証（VIP/管理者デモ版）はそのまま動作する。

## 開発者アカウント（レート制限免除）

| 用途 | Email |
|------|-------|
| オーナー | hokhok7676@gmail.com |
| DLsite検証用 | dlsite-trial@aicheckers.net |

**設定**: `backend/main.py` の `ADMIN_EMAILS`
