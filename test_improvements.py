#!/usr/bin/env python3
"""
Скрипт для тестирования улучшений Max Drawdown и Sharpe Ratio
"""
import subprocess
import json
import os

def run_backtest(search_type="dynamic"):
    """Запустить бэктест с указанным типом поиска"""
    cmd = [
        "python", "aibot2.py",
        "--symbol", "SOL/USDT",
        "--timeframe", "15m", 
        "--max-bars", "50000",
        "--search", search_type,
        "--use-dynamic-sizing",
        "--max-position-size", "0.8",
        "--dd-protection", "-0.15",
        "--turnover-cap", "0.05",
        "--max-dd-cap", "0.30"
    ]
    
    print(f"\n🔄 Запускаем бэктест с алгоритмом: {search_type}")
    print("Команда:", " ".join(cmd))
    
    try:
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
        print("✅ Бэктест завершен")
        if result.stderr:
            print("Предупреждения:", result.stderr[:500])
        return result.returncode == 0
    except subprocess.TimeoutExpired:
        print("⏰ Тайм-аут выполнения")
        return False
    except Exception as e:
        print(f"❌ Ошибка: {e}")
        return False

def compare_results():
    """Сравнить результаты оригинального и улучшенного алгоритма"""
    files = [
        ("best_thresholds.json", "Оригинальный"),
        ("best_thresholds_improved.json", "Улучшенный (пример)")
    ]
    
    print("\n📊 СРАВНЕНИЕ РЕЗУЛЬТАТОВ:")
    print("=" * 60)
    
    for filename, name in files:
        if os.path.exists(filename):
            with open(filename, 'r') as f:
                config = json.load(f)
            
            perf = config.get('performance', {})
            print(f"\n{name}:")
            print(f"  Sharpe Ratio: {perf.get('sharpe', 'N/A'):.3f}")
            print(f"  CAGR:         {perf.get('cagr', 0)*100:.1f}%")
            print(f"  Max Drawdown: {perf.get('max_dd', 0)*100:.1f}%")
            
            dynamic = config.get('dynamic_features', {})
            if dynamic:
                print(f"  Дин. размер:  {dynamic.get('use_dynamic_sizing', False)}")
                print(f"  Защита DD:    {dynamic.get('dd_protection_threshold', 'N/A')}")
        else:
            print(f"\n{name}: файл не найден")

def main():
    print("🚀 ТЕСТИРОВАНИЕ УЛУЧШЕНИЙ ТОРГОВОГО БОТА")
    print("=" * 50)
    
    # Сначала покажем текущие результаты
    print("\n1. Текущая конфигурация:")
    if os.path.exists("best_thresholds.json"):
        with open("best_thresholds.json", 'r') as f:
            current = json.load(f)
        perf = current.get('performance', {})
        print(f"   Sharpe: {perf.get('sharpe', 0):.3f}")
        print(f"   CAGR: {perf.get('cagr', 0)*100:.1f}%") 
        print(f"   Max DD: {perf.get('max_dd', 0)*100:.1f}%")
    
    # Запустим улучшенный алгоритм
    print("\n2. Запуск улучшенного алгоритма...")
    success = run_backtest("dynamic")
    
    if success:
        print("\n3. Сравнение результатов:")
        compare_results()
        
        print("\n✨ ОЖИДАЕМЫЕ УЛУЧШЕНИЯ:")
        print("  • Max Drawdown: -50% → -25-30%")
        print("  • Sharpe Ratio: 0.84 → 1.0-1.2")
        print("  • Более стабильная доходность")
        print("  • Лучшая адаптация к волатильности")
        print("  • Автоматическая защита от просадок")
    else:
        print("\n❌ Не удалось запустить улучшенный алгоритм")
        print("Попробуйте запустить вручную:")
        print("python aibot2.py --search dynamic --use-dynamic-sizing")

if __name__ == "__main__":
    main()



