#!/usr/bin/env python3
"""
Быстрый тест улучшенного алгоритма с ограниченными данными
"""
import subprocess
import sys

def run_quick_test():
    """Запуск быстрого теста с меньшим объемом данных"""
    cmd = [
        "python", "aibot2.py",
        "--symbol", "SOL/USDT",
        "--timeframe", "15m", 
        "--max-bars", "10000",  # Меньше данных для ускорения
        "--last-days", "0"  # Отключаем тест последних дней для ускорения
        # Все остальные параметры теперь оптимальны по умолчанию!
    ]
    
    print("🚀 БЫСТРЫЙ ТЕСТ улучшенного алгоритма")
    print("=" * 50)
    print("📊 Параметры:")
    print("  • Данные: 10,000 баров (~1 месяц)")
    print("  • Алгоритм: dynamic (оптимизированный)")
    print("  • Динамический размер позиции: ВКЛ")
    print("  • Защита от просадок: -15%")
    print("  • Максимальный размер позиции: 80%")
    print("\n⏱️ Ожидаемое время: 2-3 минуты...")
    print("\n" + "=" * 50)
    
    try:
        # Запускаем процесс
        process = subprocess.Popen(
            cmd, 
            stdout=subprocess.PIPE, 
            stderr=subprocess.PIPE, 
            text=True,
            bufsize=1,
            universal_newlines=True
        )
        
        # Выводим результат в реальном времени
        while True:
            output = process.stdout.readline()
            if output == '' and process.poll() is not None:
                break
            if output:
                print(output.strip())
                sys.stdout.flush()
        
        # Получаем возможные ошибки
        stderr = process.stderr.read()
        if stderr:
            print("\n⚠️ Предупреждения/ошибки:")
            print(stderr[:1000])  # Первые 1000 символов
        
        rc = process.poll()
        if rc == 0:
            print("\n✅ Быстрый тест завершен успешно!")
            print("\n📋 Результаты сохранены в:")
            print("  • best_thresholds.json - новые оптимальные пороги")
            print("  • backtest_oof_results.csv - результаты бэктеста")
            print("  • trades_all.json - все сделки")
            
            print("\n🎯 Для сравнения результатов запустите:")
            print("  python test_improvements.py")
            
        else:
            print(f"\n❌ Тест завершился с ошибкой (код: {rc})")
            
        return rc == 0
        
    except KeyboardInterrupt:
        print("\n⏹️ Тест прерван пользователем")
        if 'process' in locals():
            process.terminate()
        return False
    except Exception as e:
        print(f"\n💥 Ошибка запуска: {e}")
        return False

if __name__ == "__main__":
    success = run_quick_test()
    
    if success:
        print("\n🎉 ГОТОВО! Улучшения успешно протестированы.")
        print("\n💡 Следующие шаги:")
        print("  1. Проверьте новый best_thresholds.json")
        print("  2. Сравните результаты с оригиналом")
        print("  3. Для полного теста используйте больше данных:")
        print("     python aibot2.py --search dynamic --max-bars 50000")
    else:
        print("\n🔧 Возможные решения:")
        print("  • Проверьте интернет-соединение")
        print("  • Убедитесь, что установлены все зависимости")
        print("  • Попробуйте стандартный алгоритм:")
        print("    python aibot.py --symbol SOL/USDT --max-bars 10000")
