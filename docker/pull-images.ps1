<#
.SYNOPSIS
  下载本项目配套 Docker 镜像（etcd / minio / milvus / redis）。

.DESCRIPTION
  默认拉 CPU 版 milvus；-Gpu 拉 GPU 版。
  -Mirror 指定国内镜像前缀时，仅对 docker.io 镜像加前缀拉取，并自动 retag 回规范名，
  使 docker-compose.yml / docker-compose-gpu.yml 直接可用（无需改 compose）。
  -SaveDir 指定目录时，额外把每个镜像导出为 .tar，便于离线拷到新机器后 docker load 导入。

.EXAMPLE
  .\docker\pull-images.ps1                                   # CPU 版，官方源
  .\docker\pull-images.ps1 -Gpu                              # GPU 版 milvus
  .\docker\pull-images.ps1 -Mirror docker.1panel.live        # 走国内镜像
  .\docker\pull-images.ps1 -Gpu -Mirror docker.m.daocloud.io -SaveDir D:\imgs

.NOTES
  若 PowerShell 执行策略阻止运行，用：
    powershell -ExecutionPolicy Bypass -File .\docker\pull-images.ps1 ...
  若已为 Docker Desktop 配了代理（Settings → Resources → Proxies），则无需 -Mirror，直接官方源拉即可。
#>
param(
    [switch]$Gpu,
    [string]$Mirror = "",
    [string]$SaveDir = ""
)

$ErrorActionPreference = "Stop"

# --- 前置检查 ---
if (-not (Get-Command docker -ErrorAction SilentlyContinue)) {
    Write-Error "未找到 docker。请先安装 Docker Desktop 并启动。"; exit 1
}
docker info *> $null
if ($LASTEXITCODE -ne 0) {
    Write-Error "Docker 守护进程未运行，请先启动 Docker Desktop。"; exit 1
}

# --- 镜像清单 ---
# name: 导出 tar 的文件名前缀；image: 规范引用（compose 里用的就是它）；hub: 是否 docker.io（决定 -Mirror 是否生效）
$milvusTag = if ($Gpu) { "v2.4.10-gpu" } else { "v2.4.10" }
$images = @(
    @{ name = "etcd";   image = "quay.io/coreos/etcd:v3.5.5";                       hub = $false },
    @{ name = "minio";  image = "minio/minio:RELEASE.2023-03-20T20-16-18Z";         hub = $true  },
    @{ name = "milvus"; image = "milvusdb/milvus:$milvusTag";                        hub = $true  },
    @{ name = "redis";  image = "redis:7";                                           hub = $true  }
)

if ($SaveDir -ne "") {
    New-Item -ItemType Directory -Force -Path $SaveDir | Out-Null
}

Write-Host ("镜像版本：milvus " + $milvusTag + $(if ($Mirror) { " | 镜像源 $Mirror" } else { " | 官方源" }) + $(if ($SaveDir) { " | 导出目录 $SaveDir" } else { "" })) -ForegroundColor Yellow

# --- 逐个拉取 ---
foreach ($img in $images) {
    $canonical = $img.image
    $pullRef   = if (($Mirror -ne "") -and $img.hub) { "$Mirror/$canonical" } else { $canonical }

    Write-Host ""
    Write-Host "==> [$($img.name)] docker pull $pullRef" -ForegroundColor Cyan
    docker pull $pullRef
    if ($LASTEXITCODE -ne 0) {
        Write-Error "拉取失败：$pullRef"
        if ($Mirror -ne "" -and $img.name -eq "milvus" -and $Gpu) {
            Write-Warning "GPU milvus 的 ~1.4GB 大层在某些国内镜像上易 EOF。可换 -Mirror docker.m.daocloud.io，或开 Docker 代理后去掉 -Mirror 重跑。"
        }
        exit 1
    }

    # 走镜像源时，retag 回规范名，compose 里引用的就是规范名
    if ($pullRef -ne $canonical) {
        docker tag $pullRef $canonical | Out-Null
        Write-Host "    retag -> $canonical" -ForegroundColor DarkGray
    }

    if ($SaveDir -ne "") {
        $tar = Join-Path $SaveDir "$($img.name).tar"
        Write-Host "    docker save -> $tar" -ForegroundColor DarkGray
        docker save -o $tar $canonical
        if ($LASTEXITCODE -ne 0) { Write-Error "导出失败：$tar"; exit 1 }
    }
}

# --- 汇总 ---
Write-Host ""
Write-Host "完成。本地已就绪镜像：" -ForegroundColor Green
foreach ($img in $images) { Write-Host ("  - " + $img.image) }
Write-Host ""
Write-Host "接下来：" -ForegroundColor Green
Write-Host "  启动栈（CPU）：docker compose -f docker/docker-compose.yml up -d"
if ($Gpu) { Write-Host "  启动栈（GPU）：docker compose -f docker/docker-compose-gpu.yml up -d" }
if ($SaveDir -ne "") {
    Write-Host ""
    Write-Host "迁移到新机器：把 $SaveDir 下的 *.tar 拷过去，逐个执行：" -ForegroundColor Green
    Write-Host "  docker load -i etcd.tar   ; docker load -i minio.tar   ; docker load -i milvus.tar   ; docker load -i redis.tar"
}
