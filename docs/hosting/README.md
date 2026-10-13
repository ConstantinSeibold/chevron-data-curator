# Running Chevron on a server

Hosted mode on a Linux box: everyone signs in with their own account and works on the same projects
over HTTPS. The steps below use Docker; [Without Docker](#without-docker) is at the end.

What the box needs: an NVIDIA GPU with 24 GB (a 12 GB card works, it is just slower), 64 GB RAM,
8+ cores, and 500 GB to 1 TB of SSD. CPU-only also runs, but making masks is then very slow.

## 1. Docker and the GPU

Ubuntu, as root once:

```bash
curl -fsSL https://get.docker.com | sh
usermod -aG docker "$USER"                     # sign out and in again afterwards
# NVIDIA driver (if `nvidia-smi` does not work yet), then the container toolkit:
ubuntu-drivers install
curl -fsSL https://nvidia.github.io/libnvidia-container/gpgkey \
  | gpg --dearmor -o /usr/share/keyrings/nvidia-container-toolkit-keyring.gpg
curl -fsSL https://nvidia.github.io/libnvidia-container/stable/deb/nvidia-container-toolkit.list \
  | sed 's#deb https://#deb [signed-by=/usr/share/keyrings/nvidia-container-toolkit-keyring.gpg] https://#' \
  > /etc/apt/sources.list.d/nvidia-container-toolkit.list
apt-get update && apt-get install -y nvidia-container-toolkit
nvidia-ctk runtime configure --runtime=docker && systemctl restart docker
```

Check: `docker run --rm --gpus all ubuntu nvidia-smi` lists the card.

## 2. Chevron

```bash
git clone https://github.com/ConstantinSeibold/chevron-data-curator.git ~/chevron
cd ~/chevron
cp docs/hosting/env.example .env
mkdir -p /srv/chevron/data /srv/chevron/projects     # or the folders you set in .env
```

Edit `.env`: the two folders, `UID`/`GID` (`id -u`, `id -g`), and the domain or tunnel token from
step 4. On a box without an NVIDIA GPU, switch to the two CPU lines at the bottom.

Images go into the data folder, in any layout of subfolders. COCO files you
want to start from go there too. People can only pick images and files from inside it.

## 3. Build and make the first admin

```bash
docker compose build                                  # 10-20 min the first time
docker compose run --rm chevron user add NAME --admin # asks for a password
docker compose run --rm chevron models                # the device should say cuda
docker compose run --rm chevron models --download samhq_auto raddino
```

Weights are kept in the `models` volume, so they are downloaded once. The `dinov3*` models are gated:
accept their licence on Hugging Face and put a read token in `HF_TOKEN`.

## 4. Start it, and make it reachable

Pick one:

- A domain pointing at the server, ports 80 and 443 open: set `CHEVRON_DOMAIN`, then
  `docker compose --profile caddy up -d`. Caddy gets the certificate itself.
- No open ports or no public IP: create a tunnel in Cloudflare (Zero Trust > Networks > Tunnels),
  add a public hostname with service `http://chevron:7870`, put its token in `TUNNEL_TOKEN`, then
  `docker compose --profile tunnel up -d`.
- Only for yourself: `docker compose up -d`, then `ssh -L 7870:127.0.0.1:7870 server` and open
  http://localhost:7870.

Open the address and sign in. Add the others under the avatar menu > Users.

## Day to day

```bash
docker compose logs -f chevron                        # what it is doing
docker compose run --rm chevron user passwd NAME      # a forgotten password (also: role, remove, list)
git pull && docker compose build && docker compose up -d    # update; projects and accounts stay
```

Keep the profile flag you started with (`--profile caddy` or `--profile tunnel`) on every
`up`, or put `COMPOSE_PROFILES=caddy` in `.env`.

Back up the projects folder: it holds every project, the accounts and the sessions. The data folder
only needs a backup if the images exist nowhere else. A nightly copy is enough, for example
`rsync -a /srv/chevron/projects/ backup:/chevron/` from cron. To restore, stop Chevron, copy the
folder back and start it again.

## Without Docker

```bash
curl -LsSf https://astral.sh/uv/install.sh | sh
git clone https://github.com/ConstantinSeibold/chevron-data-curator.git ~/chevron && cd ~/chevron
uv venv --python 3.12 .venv && . .venv/bin/activate
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
uv pip install ".[all]"
mkdir -p ~/chevron/data ~/chevron/projects
chevron user add NAME --admin --root ~/chevron/projects
```

Then `chevron.service` here runs it as a systemd user service (the steps are at the top of the file),
and `Caddyfile` or `cloudflared.yml` puts it on the domain.
