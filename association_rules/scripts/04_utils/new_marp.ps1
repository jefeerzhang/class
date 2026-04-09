# 创建新的 Marp 幻灯片
param(
    [string]$name = "new_slides"
)

$templatePath = "关联算法\docs\marp_template.md"
$outputPath = "关联算法\docs\$name.md"

if (Test-Path $templatePath) {
    Copy-Item $templatePath $outputPath
    Write-Host "已创建幻灯片模板：$outputPath" -ForegroundColor Green
    Write-Host "请编辑文件修改标题和内容" -ForegroundColor Yellow
} else {
    Write-Host "错误：找不到模板文件 $templatePath" -ForegroundColor Red
}
