# Deploy LEGO Upload to Vercel + Custom Domain

## Prerequisites

- [Vercel account](https://vercel.com/signup)
- Git repo with this project (optional, for CI/CD)
- Domain name (for custom domain)

---

## 1. Deploy to Vercel

### Option A: Deploy from project folder (CLI)

```bash
# Install Vercel CLI
npm i -g vercel

# Go to project
cd lego_measure

# Deploy (first time will prompt for login / project setup)
vercel
```

Follow prompts:
- **Set up and deploy?** Yes
- **Which scope?** Your account
- **Link to existing project?** No (create new)
- **Project name?** `lego-measure` (or your choice)
- **Directory?** `./` (current folder)

After deploy you’ll get a URL like `lego-measure-xxx.vercel.app`.

### Option B: Deploy from GitHub

1. Push code to GitHub
2. Go to [vercel.com/new](https://vercel.com/new)
3. **Import** your GitHub repo
4. Set **Root Directory** to `lego_measure` (if app is in a subfolder)
5. Click **Deploy**

---

## 2. Enable Vercel Blob (for uploads)

Uploads are stored in Vercel Blob when a token is configured.

1. Open your project in [Vercel Dashboard](https://vercel.com/dashboard)
2. Go to **Storage** → **Create Database** → **Blob**
3. Name the store → **Create**
4. Keep **Private** or choose **Public**
5. The `BLOB_READ_WRITE_TOKEN` env var is set automatically for this project

Redeploy once if you added Blob after the first deploy.

---

## 3. Add a custom domain

1. In the Vercel project, go to **Settings** → **Domains**
2. Click **Add**
3. Enter your domain (e.g. `lego.yourdomain.com` or `yourdomain.com`)
4. Vercel shows DNS records:

   For subdomain (e.g. `lego.yourdomain.com`):
   - Type: **CNAME**
   - Name: `lego` (or the subdomain)
   - Value: `cname.vercel-dns.com`

   For apex domain (`yourdomain.com`):
   - Type: **A**
   - Name: `@`
   - Value: `76.76.21.21`

5. Add these records at your domain registrar (GoDaddy, Namecheap, Cloudflare, etc.)
6. Wait for DNS propagation (often 5–30 minutes, up to 48 hours)
7. Vercel will mark the domain **Valid** when DNS is correct
8. SSL certificates are issued automatically

### Domain providers (examples)

| Provider   | Where to add records          |
|-----------|--------------------------------|
| Cloudflare| DNS → Records                 |
| GoDaddy   | DNS Management → Records      |
| Namecheap | Advanced DNS → Host Records   |
| Google Domains | DNS → Custom records  |

---

## 4. Environment variables (optional)

For local Blob testing:

```bash
vercel env pull
```

This writes `.env.local` with `BLOB_READ_WRITE_TOKEN`.

---

## 5. Redeploy after changes

**Via CLI:**
```bash
cd lego_measure
vercel --prod
```

**Via GitHub:**
Commits to the main branch trigger new deploys.

---

## Summary checklist

- [ ] Deploy with `vercel` or GitHub import
- [ ] Create Blob store in project Storage
- [ ] Add domain in Settings → Domains
- [ ] Add CNAME/A record at registrar
- [ ] Open your custom URL to test uploads
