import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import itertools
from datetime import datetime, timedelta
import warnings
warnings.filterwarnings('ignore')

class ShortTradingSimulator:
    def __init__(self, data, params):
        self.data = data.copy()
        self.params = params
        self.trades = []
        self.equity_curve = []
        self.current_position = None
        
    def simulate_trades(self):
        """Основна функція симуляції торгівлі"""
        self.trades = []
        self.equity_curve = []
        balance = self.params['initial_balance']
        
        for i in range(len(self.data)):
            current_candle = self.data.iloc[i]
            
            # Перевірка сигналів входу
            if self.current_position is None:
                entry_signal = self.check_entry_signal(i)
                if entry_signal:
                    self.open_short_position(current_candle, balance)
            
            # Обробка поточної позиції
            if self.current_position is not None:
                self.update_position(current_candle, i)
                
            # Записуємо поточний стан балансу
            current_equity = balance
            if self.current_position is not None:
                unrealized_pnl = self.calculate_unrealized_pnl(current_candle['close'])
                current_equity = balance + unrealized_pnl
                
            self.equity_curve.append({
                'datetime': current_candle['datetime'],
                'equity': current_equity,
                'price': current_candle['close']
            })
        
        # Закриваємо позицію в кінці, якщо вона відкрита
        if self.current_position is not None:
            self.close_position(self.data.iloc[-1], len(self.data)-1, 'end_of_data')
            
        return self.trades, self.equity_curve
    
    def check_entry_signal(self, index):
        """Перевірка сигналу входу в шорт позицію"""
        if index < self.params['rsi_period']:
            return False
            
        # RSI перекупленість
        rsi = self.calculate_rsi(index)
        if rsi < self.params['rsi_overbought']:
            return False
            
        # Перевірка на локальний максимум
        if index < self.params['lookback_period']:
            return False
            
        current_high = self.data.iloc[index]['high']
        lookback_highs = self.data.iloc[index-self.params['lookback_period']:index]['high']
        
        if current_high <= lookback_highs.max():
            return False
            
        return True
    
    def calculate_rsi(self, index):
        """Розрахунок RSI"""
        period = self.params['rsi_period']
        if index < period:
            return 50
            
        closes = self.data.iloc[index-period:index+1]['close']
        delta = closes.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        
        rs = gain.iloc[-1] / loss.iloc[-1] if loss.iloc[-1] != 0 else 0
        rsi = 100 - (100 / (1 + rs))
        return rsi
    
    def open_short_position(self, candle, balance):
        """Відкриття шорт позиції"""
        entry_price = candle['close']
        position_size = balance * self.params['position_size_pct'] / 100
        quantity = position_size / entry_price
        
        # Розрахунок рівнів
        stop_loss = entry_price * (1 + self.params['stop_loss_pct'] / 100)
        take_profit = entry_price * (1 - self.params['take_profit_pct'] / 100)
        breakeven_price = entry_price
        
        self.current_position = {
            'entry_price': entry_price,
            'quantity': quantity,
            'entry_time': candle['datetime'],
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'breakeven_price': breakeven_price,
            'is_breakeven_moved': False,
            'averaging_count': 0,
            'total_quantity': quantity,
            'avg_entry_price': entry_price
        }
    
    def update_position(self, candle, index):
        """Оновлення поточної позиції"""
        current_price = candle['close']
        high_price = candle['high']
        low_price = candle['low']
        
        # Перевірка стоп-лоссу
        if high_price >= self.current_position['stop_loss']:
            self.close_position(candle, index, 'stop_loss')
            return
            
        # Перевірка тейк-профіту
        if low_price <= self.current_position['take_profit']:
            self.close_position(candle, index, 'take_profit')
            return
        
        # Логіка усереднення
        if self.should_average_down(current_price):
            self.add_to_position(candle)
        
        # Переміщення до брейк-івену
        if not self.current_position['is_breakeven_moved']:
            profit_pct = (self.current_position['avg_entry_price'] - current_price) / self.current_position['avg_entry_price'] * 100
            if profit_pct >= self.params['breakeven_trigger_pct']:
                self.current_position['stop_loss'] = self.current_position['avg_entry_price']
                self.current_position['is_breakeven_moved'] = True
    
    def should_average_down(self, current_price):
        """Перевірка чи потрібно усереднюватися"""
        if self.current_position['averaging_count'] >= self.params['max_averaging']:
            return False
            
        price_change = (current_price - self.current_position['avg_entry_price']) / self.current_position['avg_entry_price'] * 100
        
        if price_change >= self.params['averaging_threshold_pct']:
            return True
            
        return False
    
    def add_to_position(self, candle):
        """Додавання до позиції (усереднення)"""
        entry_price = candle['close']
        additional_size = self.params['initial_balance'] * self.params['position_size_pct'] / 100 * self.params['averaging_multiplier']
        additional_quantity = additional_size / entry_price
        
        # Оновлюємо середню ціну входу
        total_cost = (self.current_position['total_quantity'] * self.current_position['avg_entry_price'] + 
                     additional_quantity * entry_price)
        self.current_position['total_quantity'] += additional_quantity
        self.current_position['avg_entry_price'] = total_cost / self.current_position['total_quantity']
        
        # Оновлюємо стоп-лосс відносно нової середньої ціни
        self.current_position['stop_loss'] = self.current_position['avg_entry_price'] * (1 + self.params['stop_loss_pct'] / 100)
        self.current_position['averaging_count'] += 1
    
    def calculate_unrealized_pnl(self, current_price):
        """Розрахунок нереалізованого PnL"""
        if self.current_position is None:
            return 0
            
        price_diff = self.current_position['avg_entry_price'] - current_price
        return price_diff * self.current_position['total_quantity']
    
    def close_position(self, candle, index, reason):
        """Закриття позиції"""
        exit_price = candle['close']
        
        if reason == 'stop_loss':
            exit_price = self.current_position['stop_loss']
        elif reason == 'take_profit':
            exit_price = self.current_position['take_profit']
            
        pnl = (self.current_position['avg_entry_price'] - exit_price) * self.current_position['total_quantity']
        pnl_pct = pnl / (self.current_position['avg_entry_price'] * self.current_position['total_quantity']) * 100
        
        trade_record = {
            'entry_time': self.current_position['entry_time'],
            'exit_time': candle['datetime'],
            'entry_price': self.current_position['avg_entry_price'],
            'exit_price': exit_price,
            'quantity': self.current_position['total_quantity'],
            'pnl': pnl,
            'pnl_pct': pnl_pct,
            'reason': reason,
            'averaging_count': self.current_position['averaging_count'],
            'duration_minutes': index - self.data[self.data['datetime'] == self.current_position['entry_time']].index[0]
        }
        
        self.trades.append(trade_record)
        self.current_position = None

def analyze_multiple_coins(data, selected_coins, params):
    """Аналіз множинних монет"""
    all_results = {}
    
    for coin in selected_coins:
        coin_data = data[data['coin'] == coin].copy().reset_index(drop=True)
        
        if len(coin_data) < 50:  # Мінімальна кількість даних
            continue
            
        simulator = ShortTradingSimulator(coin_data, params)
        trades, equity_curve = simulator.simulate_trades()
        metrics = calculate_metrics(trades, equity_curve)
        
        all_results[coin] = {
            'trades': trades,
            'equity_curve': equity_curve,
            'metrics': metrics,
            'data': coin_data
        }
    
    return all_results

def optimize_parameters_multiple_coins(data, selected_coins, param_ranges):
    """Оптимізація параметрів для множинних монет"""
    
    # Створюємо всі можливі комбінації параметрів
    param_names = list(param_ranges.keys())
    param_values = list(param_ranges.values())
    
    all_combinations = list(itertools.product(*param_values))
    
    results = []
    progress_bar = st.progress(0)
    status_text = st.empty()
    
    for i, combination in enumerate(all_combinations):
        params = dict(zip(param_names, combination))
        
        # Додаємо фіксовані параметри
        params.update({
            'initial_balance': 1000,
            'lookback_period': 5,
            'breakeven_trigger_pct': 1.0
        })
        
        # Аналізуємо всі монети з цими параметрами
        coin_results = []
        
        for coin in selected_coins:
            coin_data = data[data['coin'] == coin].copy().reset_index(drop=True)
            
            if len(coin_data) < 50:
                continue
                
            try:
                simulator = ShortTradingSimulator(coin_data, params)
                trades, equity_curve = simulator.simulate_trades()
                metrics = calculate_metrics(trades, equity_curve)
                
                if metrics and metrics.get('total_trades', 0) > 0:
                    coin_result = {**params, **metrics, 'coin': coin}
                    coin_results.append(coin_result)
            except Exception as e:
                continue
        
        # Додаємо результати по монетах
        results.extend(coin_results)
        
        # Оновлюємо прогрес
        progress = (i + 1) / len(all_combinations)
        progress_bar.progress(progress)
        status_text.text(f'Оброблено {i+1}/{len(all_combinations)} комбінацій...')
    
    progress_bar.empty()
    status_text.empty()
    
    return pd.DataFrame(results)

def calculate_metrics(trades, equity_curve):
    """Розрахунок торгових метрик"""
    if not trades:
        return {}
        
    df_trades = pd.DataFrame(trades)
    
    total_trades = len(trades)
    winning_trades = len(df_trades[df_trades['pnl'] > 0])
    losing_trades = len(df_trades[df_trades['pnl'] < 0])
    
    win_rate = winning_trades / total_trades * 100 if total_trades > 0 else 0
    
    total_pnl = df_trades['pnl'].sum()
    avg_pnl_per_trade = df_trades['pnl'].mean()
    
    if winning_trades > 0:
        avg_win = df_trades[df_trades['pnl'] > 0]['pnl'].mean()
    else:
        avg_win = 0
        
    if losing_trades > 0:
        avg_loss = df_trades[df_trades['pnl'] < 0]['pnl'].mean()
    else:
        avg_loss = 0
        
    profit_factor = abs(avg_win * winning_trades / (avg_loss * losing_trades)) if losing_trades > 0 and avg_loss != 0 else float('inf')
    
    # Розрахунок максимальної просадки
    df_equity = pd.DataFrame(equity_curve)
    peak = df_equity['equity'].expanding().max()
    drawdown = (df_equity['equity'] - peak) / peak * 100
    max_drawdown = drawdown.min()
    
    return {
        'total_trades': total_trades,
        'win_rate': win_rate,
        'total_pnl': total_pnl,
        'avg_pnl_per_trade': avg_pnl_per_trade,
        'avg_win': avg_win,
        'avg_loss': avg_loss,
        'profit_factor': profit_factor,
        'max_drawdown': max_drawdown,
        'winning_trades': winning_trades,
        'losing_trades': losing_trades
    }

def display_single_coin_results(trades, equity_curve, metrics, data, params):
    """Відображення результатів для однієї монети"""
    # Показуємо метрики
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("Win Rate", f"{metrics['win_rate']:.1f}%")
        st.metric("Угод всього", f"{metrics['total_trades']}")
    
    with col2:
        st.metric("Total PnL", f"{metrics['total_pnl']:.2f}")
        st.metric("PnL за угоду", f"{metrics['avg_pnl_per_trade']:.2f}")
    
    with col3:
        st.metric("Середній виграш", f"{metrics['avg_win']:.2f}")
        st.metric("Середній програш", f"{metrics['avg_loss']:.2f}")
    
    with col4:
        st.metric("Profit Factor", f"{metrics['profit_factor']:.2f}")
        st.metric("Max Drawdown", f"{metrics['max_drawdown']:.2f}%")
    
    # Графік equity curve
    st.subheader("📈 Крива капіталу")
    
    df_equity = pd.DataFrame(equity_curve)
    
    fig_equity = make_subplots(
        rows=2, cols=1,
        subplot_titles=('Крива капіталу', 'Ціна активу'),
        vertical_spacing=0.1
    )
    
    fig_equity.add_trace(
        go.Scatter(
            x=df_equity['datetime'],
            y=df_equity['equity'],
            name='Equity',
            line=dict(color='blue')
        ),
        row=1, col=1
    )
    
    fig_equity.add_trace(
        go.Scatter(
            x=df_equity['datetime'],
            y=df_equity['price'],
            name='Price',
            line=dict(color='orange')
        ),
        row=2, col=1
    )
    
    fig_equity.update_layout(height=600, showlegend=True)
    st.plotly_chart(fig_equity, use_container_width=True)
    
    # Таблиця угод
    st.subheader("📋 Детальна інформація по угодам")
    
    df_trades = pd.DataFrame(trades)
    df_trades['pnl_color'] = df_trades['pnl'].apply(lambda x: '🟢' if x > 0 else '🔴')
    
    # Форматування для відображення
    display_df = df_trades.copy()
    display_df['entry_time'] = pd.to_datetime(display_df['entry_time']).dt.strftime('%Y-%m-%d %H:%M')
    display_df['exit_time'] = pd.to_datetime(display_df['exit_time']).dt.strftime('%Y-%m-%d %H:%M')
    display_df = display_df.round(4)
    
    st.dataframe(
        display_df[['pnl_color', 'entry_time', 'exit_time', 'entry_price', 
                  'exit_price', 'pnl', 'pnl_pct', 'reason', 'averaging_count']],
        use_container_width=True
    )
    
    # Додаткові графіки аналізу
    col1, col2 = st.columns(2)
    
    with col1:
        # Розподіл PnL (виправлена версія)
        fig_pnl_dist = px.histogram(
            df_trades, 
            x='pnl_pct', 
            title='Розподіл PnL (%)',
            nbins=20,
            color_discrete_sequence=['lightblue']
        )
        fig_pnl_dist.add_vline(x=0, line_dash="dash", line_color="red")
        st.plotly_chart(fig_pnl_dist, use_container_width=True)
    
    with col2:
        # PnL по причинах закриття
        reason_stats = df_trades.groupby('reason')['pnl'].agg(['count', 'sum', 'mean']).round(2)
        reason_stats.columns = ['Кількість', 'Загальний PnL', 'Середній PnL']
        st.write("**Статистика по причинах закриття:**")
        st.dataframe(reason_stats)
    
    # Графік позначок входу/виходу на ціновому графіку
    st.subheader("📍 Позначки угод на ціновому графіку")
    
    fig_price = go.Figure()
    
    # Додаємо лінію ціни
    fig_price.add_trace(
        go.Scatter(
            x=data['datetime'],
            y=data['close'],
            name='Ціна закриття',
            line=dict(color='black', width=1)
        )
    )
    
    # Додаємо точки входу (червоні)
    entry_points = []
    exit_points = []
    
    for trade in trades:
        entry_idx = data[data['datetime'] == trade['entry_time']].index
        exit_idx = data[data['datetime'] == trade['exit_time']].index
        
        if len(entry_idx) > 0:
            entry_points.append({
                'datetime': trade['entry_time'],
                'price': trade['entry_price'],
                'type': 'Вхід в шорт'
            })
        
        if len(exit_idx) > 0:
            exit_points.append({
                'datetime': trade['exit_time'],
                'price': trade['exit_price'],
                'type': f'Вихід ({trade["reason"]})',
                'color': 'green' if trade['pnl'] > 0 else 'red'
            })
    
    if entry_points:
        df_entry = pd.DataFrame(entry_points)
        fig_price.add_trace(
            go.Scatter(
                x=df_entry['datetime'],
                y=df_entry['price'],
                mode='markers',
                name='Вхід в шорт',
                marker=dict(color='red', size=8, symbol='triangle-down')
            )
        )
    
    if exit_points:
        df_exit = pd.DataFrame(exit_points)
        
        # Прибуткові виходи
        profitable_exits = df_exit[df_exit['color'] == 'green']
        if len(profitable_exits) > 0:
            fig_price.add_trace(
                go.Scatter(
                    x=profitable_exits['datetime'],
                    y=profitable_exits['price'],
                    mode='markers',
                    name='Прибутковий вихід',
                    marker=dict(color='green', size=8, symbol='triangle-up')
                )
            )
        
        # Збиткові виходи
        losing_exits = df_exit[df_exit['color'] == 'red']
        if len(losing_exits) > 0:
            fig_price.add_trace(
                go.Scatter(
                    x=losing_exits['datetime'],
                    y=losing_exits['price'],
                    mode='markers',
                    name='Збитковий вихід',
                    marker=dict(color='darkred', size=8, symbol='triangle-up')
                )
            )
    
    fig_price.update_layout(
        title='Ціновий графік з позначками угод',
        xaxis_title='Час',
        yaxis_title='Ціна',
        height=500
    )
    
    st.plotly_chart(fig_price, use_container_width=True)
    
    # Часова статистика
    st.subheader("⏰ Часова статистика")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        avg_duration = df_trades['duration_minutes'].mean()
        st.metric("Середня тривалість угоди", f"{avg_duration:.0f} хв")
    
    with col2:
        profitable_duration = df_trades[df_trades['pnl'] > 0]['duration_minutes'].mean()
        if not pd.isna(profitable_duration):
            st.metric("Тривалість прибуткових угод", f"{profitable_duration:.0f} хв")
        else:
            st.metric("Тривалість прибуткових угод", "N/A")
    
    with col3:
        losing_duration = df_trades[df_trades['pnl'] < 0]['duration_minutes'].mean()
        if not pd.isna(losing_duration):
            st.metric("Тривалість збиткових угод", f"{losing_duration:.0f} хв")
        else:
            st.metric("Тривалість збиткових угод", "N/A")
    
    # Розподіл тривалості угод (виправлена версія)
    fig_duration = px.histogram(
        df_trades,
        x='duration_minutes',
        title='Розподіл тривалості угод (хвилини)',
        nbins=20,
        color_discrete_sequence=['lightcoral']
    )
    st.plotly_chart(fig_duration, use_container_width=True)
    
    # Експорт результатів
    st.subheader("📤 Експорт результатів")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("📋 Копіювати параметри"):
            st.code(str(params), language='python')
    
    with col2:
        # CSV експорт угод
        csv_data = df_trades.to_csv(index=False)
        st.download_button(
            label="💾 Завантажити угоди (CSV)",
            data=csv_data,
            file_name=f"short_strategy_trades_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
            mime="text/csv"
        )

def display_multiple_coins_results(all_results, selected_coins):
    """Відображення результатів для множинних монет"""
    
    # Загальна статистика по монетах
    summary_data = []
    all_trades = []
    
    for coin, results in all_results.items():
        if results['metrics']:
            summary_data.append({
                'Монета': coin,
                'Угод': results['metrics']['total_trades'],
                'Win Rate (%)': round(results['metrics']['win_rate'], 1),
                'Total PnL': round(results['metrics']['total_pnl'], 2),
                'Max Drawdown (%)': round(results['metrics']['max_drawdown'], 2),
                'Profit Factor': round(results['metrics']['profit_factor'], 2)
            })
            # Додаємо інформацію про монету до кожної угоди
            for trade in results['trades']:
                trade_copy = trade.copy()
                trade_copy['coin'] = coin
                all_trades.append(trade_copy)
    
    if summary_data:
        st.subheader("📊 Зведення по монетах")
        df_summary = pd.DataFrame(summary_data)
        st.dataframe(df_summary, use_container_width=True)
        
        # Загальні метрики
        col1, col2, col3, col4 = st.columns(4)
        
        total_trades = sum([r['metrics']['total_trades'] for r in all_results.values() if r['metrics']])
        avg_win_rate = np.mean([r['metrics']['win_rate'] for r in all_results.values() if r['metrics']])
        total_pnl = sum([r['metrics']['total_pnl'] for r in all_results.values() if r['metrics']])
        avg_drawdown = np.mean([r['metrics']['max_drawdown'] for r in all_results.values() if r['metrics']])
        
        with col1:
            st.metric("Всього угод", f"{total_trades}")
        with col2:
            st.metric("Середній Win Rate", f"{avg_win_rate:.1f}%")
        with col3:
            st.metric("Загальний PnL", f"{total_pnl:.2f}")
        with col4:
            st.metric("Середній Drawdown", f"{avg_drawdown:.2f}%")
        
        # Графік порівняння монет
        st.subheader("📈 Порівняння результатів по монетах")
        
        fig_comparison = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Win Rate по монетах', 'Total PnL по монетах', 
                          'Кількість угод', 'Max Drawdown'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )
        
        coins = [item['Монета'] for item in summary_data]
        win_rates = [item['Win Rate (%)'] for item in summary_data]
        pnls = [item['Total PnL'] for item in summary_data]
        trades_count = [item['Угод'] for item in summary_data]
        drawdowns = [abs(item['Max Drawdown (%)']) for item in summary_data]
        
        fig_comparison.add_trace(
            go.Bar(x=coins, y=win_rates, name="Win Rate", marker_color='lightblue'),
            row=1, col=1
        )
        
        fig_comparison.add_trace(
            go.Bar(x=coins, y=pnls, name="PnL", marker_color='lightgreen'),
            row=1, col=2
        )
        
        fig_comparison.add_trace(
            go.Bar(x=coins, y=trades_count, name="Угоди", marker_color='orange'),
            row=2, col=1
        )
        
        fig_comparison.add_trace(
            go.Bar(x=coins, y=drawdowns, name="Drawdown", marker_color='lightcoral'),
            row=2, col=2
        )
        
        fig_comparison.update_layout(height=800, showlegend=False)
        st.plotly_chart(fig_comparison, use_container_width=True)
        
        # Детальна таблиця всіх угод
        if all_trades:
            st.subheader("📋 Всі угоди по монетах")
            
            df_all_trades = pd.DataFrame(all_trades)
            df_all_trades['pnl_color'] = df_all_trades['pnl'].apply(lambda x: '🟢' if x > 0 else '🔴')
            
            # Форматування
            display_df = df_all_trades.copy()
            display_df['entry_time'] = pd.to_datetime(display_df['entry_time']).dt.strftime('%Y-%m-%d %H:%M')
            display_df['exit_time'] = pd.to_datetime(display_df['exit_time']).dt.strftime('%Y-%m-%d %H:%M')
            display_df = display_df.round(4)
            
            # Фільтр по монетах
            selected_coin_filter = st.selectbox(
                "Фільтр по монеті (опціонально):",
                ["Всі монети"] + list(selected_coins)
            )
            
            if selected_coin_filter != "Всі монети":
                display_df = display_df[display_df['coin'] == selected_coin_filter]
            
            st.dataframe(
                display_df[['coin', 'pnl_color', 'entry_time', 'exit_time', 'entry_price', 
                          'exit_price', 'pnl', 'pnl_pct', 'reason', 'averaging_count']],
                use_container_width=True
            )
            
            # Експорт
            st.subheader("📤 Експорт результатів")
            csv_data = df_all_trades.to_csv(index=False)
            st.download_button(
                label="💾 Завантажити всі угоди (CSV)",
                data=csv_data,
                file_name=f"short_strategy_all_coins_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                mime="text/csv"
            )

# Streamlit інтерфейс
st.set_page_config(page_title="Crypto Short Trading Analyzer", layout="wide")
st.title("🔻 Аналізатор шорт-стратегій криптовалют")

# Завантаження файлу
st.sidebar.header("📁 Завантаження даних")
uploaded_file = st.sidebar.file_uploader(
    "Завантажте CSV файл з історичними даними", 
    type=['csv'],
    help="Формат: coin;volume;open;high;low;close;datetime"
)

if uploaded_file is not None:
    try:
        # Завантажуємо та обробляємо дані
        data = pd.read_csv(uploaded_file, delimiter=';')
        
        # Перевіряємо наявність необхідних колонок
        required_columns = ['coin', 'volume', 'open', 'high', 'low', 'close', 'datetime']
        if not all(col in data.columns for col in required_columns):
            st.error(f"Файл повинен містити колонки: {', '.join(required_columns)}")
            st.stop()
        
        # Конвертуємо datetime з підтримкою різних форматів
        try:
            # Спробуємо автоматичне визначення з dayfirst=True для европейського формату
            data['datetime'] = pd.to_datetime(data['datetime'], dayfirst=True, format='mixed')
        except:
            try:
                # Спробуємо конкретний формат для вашого файлу
                data['datetime'] = pd.to_datetime(data['datetime'], format='%d.%m.%Y %H:%M')
            except:
                try:
                    # Альтернативні формати
                    data['datetime'] = pd.to_datetime(data['datetime'], format='%d.%m.%Y %H:%M:%S')
                except:
                    try:
                        # Загальне перетворення без конкретного формату
                        data['datetime'] = pd.to_datetime(data['datetime'], infer_datetime_format=True)
                    except Exception as e:
                        st.error(f"Не вдалося конвертувати дату. Формат у файлі: {data['datetime'].iloc[0]}. Помилка: {str(e)}")
                        st.write("Підтримувані формати дат:")
                        st.write("- DD.MM.YYYY HH:MM")
                        st.write("- DD.MM.YYYY HH:MM:SS") 
                        st.write("- YYYY-MM-DD HH:MM:SS")
                        st.write("- MM/DD/YYYY HH:MM")
                        st.stop()
        
        data = data.sort_values('datetime').reset_index(drop=True)
        
        # Конвертуємо числові колонки
        numeric_cols = ['volume', 'open', 'high', 'low', 'close']
        for col in numeric_cols:
            data[col] = pd.to_numeric(data[col], errors='coerce')
        
        # Видаляємо рядки з NaN
        data = data.dropna().reset_index(drop=True)
        
        st.sidebar.success(f"✅ Завантажено {len(data)} записів")
        
        # Показуємо базову інформацію
        st.sidebar.write(f"**Період:** {data['datetime'].min()} - {data['datetime'].max()}")
        
        # Вибір монет для аналізу
        available_coins = sorted(data['coin'].unique())
        st.sidebar.write(f"**Доступні монети:** {', '.join(available_coins)}")
        
        st.sidebar.subheader("🪙 Вибір монет для аналізу")
        
        # Опції вибору
        coin_selection_method = st.sidebar.radio(
            "Спосіб вибору:",
            ["Одна монета", "Кілька монет", "Всі монети"]
        )
        
        selected_coins = []
        
        if coin_selection_method == "Одна монета":
            selected_coin = st.sidebar.selectbox("Виберіть монету:", available_coins)
            selected_coins = [selected_coin]
        elif coin_selection_method == "Кілька монет":
            selected_coins = st.sidebar.multiselect(
                "Виберіть монети:", 
                available_coins,
                default=[available_coins[0]] if available_coins else []
            )
        else:  # Всі монети
            selected_coins = available_coins
            st.sidebar.write(f"Будуть проаналізовані всі {len(selected_coins)} монет")
        
        if not selected_coins:
            st.error("Оберіть хоча б одну монету для аналізу")
            st.stop()
        
        # Фільтруємо дані за вибраними монетами
        filtered_data = data[data['coin'].isin(selected_coins)].copy()
        
        st.sidebar.write(f"**Вибрано монет:** {len(selected_coins)}")
        st.sidebar.write(f"**Записів після фільтрації:** {len(filtered_data)}")
        
        # Налаштування параметрів
        st.sidebar.header("⚙️ Параметри стратегії")
        
        # Режим роботи
        mode = st.sidebar.selectbox(
            "Режим роботи",
            ["🔍 Оптимізація параметрів", "📊 Тестування стратегії"]
        )
        
        if mode == "🔍 Оптимізація параметрів":
            st.header("🔍 Оптимізація параметрів")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Діапазони параметрів для оптимізації")
                
                # Параметри RSI
                rsi_period_range = st.slider("RSI період", 5, 30, (10, 20), 5)
                rsi_overbought_range = st.slider("RSI перекупленість", 60, 90, (70, 85), 5)
                
                # Параметри позиції
                position_size_range = st.slider("Розмір позиції (%)", 5, 50, (10, 30), 5)
                
                # Параметри ризику
                stop_loss_range = st.slider("Стоп-лосс (%)", 2, 15, (5, 10), 1)
                take_profit_range = st.slider("Тейк-профіт (%)", 2, 15, (3, 8), 1)
            
            with col2:
                st.subheader("Параметри усереднення")
                
                max_averaging_range = st.slider("Макс. усереднень", 0, 5, (1, 3))
                averaging_threshold_range = st.slider("Поріг усереднення (%)", 5, 20, (8, 15), 2)
                averaging_multiplier_range = st.slider("Множник усереднення", 1.0, 3.0, (1.5, 2.5), 0.5)
            
            # Кнопка запуску оптимізації
            if st.button("🚀 Запустити оптимізацію", type="primary"):
                
                param_ranges = {
                    'rsi_period': list(range(rsi_period_range[0], rsi_period_range[1] + 1, 5)),
                    'rsi_overbought': list(range(rsi_overbought_range[0], rsi_overbought_range[1] + 1, 5)),
                    'position_size_pct': list(range(position_size_range[0], position_size_range[1] + 1, 5)),
                    'stop_loss_pct': list(range(stop_loss_range[0], stop_loss_range[1] + 1, 1)),
                    'take_profit_pct': list(range(take_profit_range[0], take_profit_range[1] + 1, 1)),
                    'max_averaging': list(range(max_averaging_range[0], max_averaging_range[1] + 1)),
                    'averaging_threshold_pct': list(range(averaging_threshold_range[0], averaging_threshold_range[1] + 1, 2)),
                    'averaging_multiplier': [x/10 for x in range(int(averaging_multiplier_range[0]*10), int(averaging_multiplier_range[1]*10) + 1, 5)]
                }
                
                with st.spinner("Виконується оптимізація..."):
                    results_df = optimize_parameters_multiple_coins(filtered_data, selected_coins, param_ranges)
                
                if not results_df.empty:
                    st.success(f"✅ Знайдено {len(results_df)} успішних конфігурацій по {len(selected_coins)} монетах")
                    
                    # Показуємо результати по монетах
                    st.subheader("🪙 Результати по монетах")
                    coin_summary = results_df.groupby('coin').agg({
                        'win_rate': 'mean',
                        'total_pnl': 'mean', 
                        'total_trades': 'mean',
                        'max_drawdown': 'mean'
                    }).round(2)
                    st.dataframe(coin_summary, use_container_width=True)
                    
                    # Топ результати (загальні)
                    col1, col2, col3 = st.columns(3)
                    
                    with col1:
                        st.subheader("🏆 Найвищий Win Rate")
                        best_winrate = results_df.loc[results_df['win_rate'].idxmax()]
                        st.metric("Win Rate", f"{best_winrate['win_rate']:.1f}%")
                        st.metric("Total PnL", f"{best_winrate['total_pnl']:.2f}")
                        st.metric("Угод", f"{best_winrate['total_trades']:.0f}")
                        st.metric("Монета", f"{best_winrate['coin']}")
                        
                        with st.expander("Параметри"):
                            for key, value in best_winrate.items():
                                if key.endswith('_pct') or key.endswith('_period') or key == 'max_averaging' or key == 'averaging_multiplier':
                                    st.write(f"**{key}**: {value}")
                    
                    with col2:
                        st.subheader("💰 Найвищий PnL")
                        best_pnl = results_df.loc[results_df['total_pnl'].idxmax()]
                        st.metric("Total PnL", f"{best_pnl['total_pnl']:.2f}")
                        st.metric("Win Rate", f"{best_pnl['win_rate']:.1f}%")
                        st.metric("Угод", f"{best_pnl['total_trades']:.0f}")
                        st.metric("Монета", f"{best_pnl['coin']}")
                        
                        with st.expander("Параметри"):
                            for key, value in best_pnl.items():
                                if key.endswith('_pct') or key.endswith('_period') or key == 'max_averaging' or key == 'averaging_multiplier':
                                    st.write(f"**{key}**: {value}")
                    
                    with col3:
                        st.subheader("🛡️ Найменший Drawdown")
                        best_dd = results_df.loc[results_df['max_drawdown'].idxmax()]  # Найменша просадка (найбільше значення, бо негативне)
                        st.metric("Max Drawdown", f"{best_dd['max_drawdown']:.2f}%")
                        st.metric("Total PnL", f"{best_dd['total_pnl']:.2f}")
                        st.metric("Win Rate", f"{best_dd['win_rate']:.1f}%")
                        st.metric("Монета", f"{best_dd['coin']}")
                        
                        with st.expander("Параметри"):
                            for key, value in best_dd.items():
                                if key.endswith('_pct') or key.endswith('_period') or key == 'max_averaging' or key == 'averaging_multiplier':
                                    st.write(f"**{key}**: {value}")
                    
                    # Heatmap результатів
                    st.subheader("📊 Heatmap результатів")
                    
                    metric_choice = st.selectbox(
                        "Виберіть метрику для відображення",
                        ["win_rate", "total_pnl", "max_drawdown", "profit_factor"]
                    )
                    
                    # Створюємо pivot table для heatmap
                    pivot_data = results_df.pivot_table(
                        values=metric_choice,
                        index='stop_loss_pct',
                        columns='take_profit_pct',
                        aggfunc='mean'
                    )
                    
                    fig_heatmap = px.imshow(
                        pivot_data,
                        labels=dict(x="Take Profit %", y="Stop Loss %", color=metric_choice),
                        title=f"Heatmap: {metric_choice}",
                        aspect="auto"
                    )
                    st.plotly_chart(fig_heatmap, use_container_width=True)
                    
                    # Таблиця всіх результатів
                    st.subheader("📋 Детальні результати")
                    
                    # Сортування результатів
                    sort_by = st.selectbox(
                        "Сортувати за:",
                        ["win_rate", "total_pnl", "max_drawdown", "profit_factor"],
                        key="sort_results"
                    )
                    
                    results_display = results_df.sort_values(sort_by, ascending=False).head(50)
                    
                    # Форматуємо для відображення
                    display_cols = [
                        'coin', 'win_rate', 'total_pnl', 'total_trades', 'max_drawdown', 
                        'profit_factor', 'rsi_period', 'rsi_overbought', 
                        'position_size_pct', 'stop_loss_pct', 'take_profit_pct',
                        'max_averaging', 'averaging_threshold_pct', 'averaging_multiplier'
                    ]
                    
                    st.dataframe(
                        results_display[display_cols].round(2),
                        use_container_width=True
                    )
                    
                else:
                    st.error("Не знайдено успішних конфігурацій. Спробуйте розширити діапазони параметрів.")
        
        else:  # Режим тестування стратегії
            st.header("📊 Тестування стратегії")
            
            col1, col2 = st.columns(2)
            
            with col1:
                st.subheader("Параметри стратегії")
                
                rsi_period = st.slider("RSI період", 5, 30, 14)
                rsi_overbought = st.slider("RSI перекупленість", 60, 90, 75)
                position_size_pct = st.slider("Розмір позиції (%)", 5, 50, 20)
                stop_loss_pct = st.slider("Стоп-лосс (%)", 2, 15, 7)
                take_profit_pct = st.slider("Тейк-профіт (%)", 2, 15, 5)
            
            with col2:
                st.subheader("Параметри усереднення")
                
                max_averaging = st.slider("Максимум усереднень", 0, 5, 2)
                averaging_threshold_pct = st.slider("Поріг усереднення (%)", 5, 20, 10)
                averaging_multiplier = st.slider("Множник усереднення", 1.0, 3.0, 2.0, 0.1)
            
            # Кнопка запуску тестування
            if st.button("🚀 Запустити тест", type="primary"):
                
                params = {
                    'initial_balance': 1000,
                    'rsi_period': rsi_period,
                    'rsi_overbought': rsi_overbought,
                    'position_size_pct': position_size_pct,
                    'stop_loss_pct': stop_loss_pct,
                    'take_profit_pct': take_profit_pct,
                    'max_averaging': max_averaging,
                    'averaging_threshold_pct': averaging_threshold_pct,
                    'averaging_multiplier': averaging_multiplier,
                    'lookback_period': 5,
                    'breakeven_trigger_pct': 1.0
                }
                
                with st.spinner("Виконується симуляція..."):
                    if len(selected_coins) == 1:
                        # Аналіз однієї монети
                        coin_data = filtered_data.copy().reset_index(drop=True)
                        simulator = ShortTradingSimulator(coin_data, params)
                        trades, equity_curve = simulator.simulate_trades()
                        metrics = calculate_metrics(trades, equity_curve)
                        
                        # Показуємо результати для однієї монети
                        if trades:
                            st.subheader(f"📊 Результати для {selected_coins[0]}")
                            display_single_coin_results(trades, equity_curve, metrics, coin_data, params)
                        else:
                            st.warning("⚠️ За заданими параметрами угоди не були знайдені для цієї монети.")
                    else:
                        # Аналіз множинних монет
                        all_results = analyze_multiple_coins(filtered_data, selected_coins, params)
                        
                        if all_results:
                            st.subheader(f"📊 Результати по {len(selected_coins)} монетах")
                            display_multiple_coins_results(all_results, selected_coins)
                        else:
                            st.warning("⚠️ За заданими параметрами угоди не були знайдені для жодної монети.")
    
    except Exception as e:
        st.error(f"❌ Помилка при обробці файлу: {str(e)}")
        st.write("**Перевірте, що файл має правильний формат:**")
        st.write("- Розділювач: точка з комою (;)")
        st.write("- Колонки: coin;volume;open;high;low;close;datetime")
        st.write("- Формат datetime: DD.MM.YYYY HH:MM або YYYY-MM-DD HH:MM:SS")
        st.write("- Приклад: BTC;1000.5;45000;45200;44900;45100;13.07.2025 00:00")

else:
    st.info("👆 Завантажте CSV файл для початку роботи")
    
    # Показуємо приклад формату файлу
    st.subheader("📋 Приклад формату даних")
    
    example_data = pd.DataFrame({
        'coin': ['BTC', 'BTC', 'BTC', 'BTC', 'BTC'],
        'volume': [1000.5, 1200.3, 950.8, 1100.2, 980.7],
        'open': [45000.0, 45100.0, 44950.0, 45050.0, 45200.0],
        'high': [45200.0, 45300.0, 45100.0, 45250.0, 45400.0],
        'low': [44900.0, 44800.0, 44850.0, 44950.0, 45000.0],
        'close': [45100.0, 44950.0, 45050.0, 45200.0, 45150.0],
        'datetime': ['13.07.2025 00:00', '13.07.2025 00:01', '13.07.2025 00:02', '13.07.2025 00:03', '13.07.2025 00:04']
    })
    
    st.dataframe(example_data, use_container_width=True)
    
    st.write("**Підтримувані формати дат:**")
    st.write("✅ `13.07.2025 00:00` (DD.MM.YYYY HH:MM)")
    st.write("✅ `13.07.2025 00:00:00` (DD.MM.YYYY HH:MM:SS)")
    st.write("✅ `2025-07-13 00:00:00` (YYYY-MM-DD HH:MM:SS)")
    st.write("✅ `07/13/2025 00:00` (MM/DD/YYYY HH:MM)")
    
    st.write("**Опис стратегії:**")
    st.write("🔻 **Шорт-стратегія** - це торгова стратегія, яка заробляє на падінні цін активів")
    st.write("📊 **RSI** - індикатор відносної сили, який допомагає визначити перекупленість")
    st.write("⚖️ **Усереднення** - додавання до позиції при несприятливому русі ціни")
    st.write("🛡️ **Брейк-івен** - переміщення стоп-лоссу в точку беззбитковості при прибутку")
    st.write("📈 **Grid Search** - автоматичний пошук найкращих параметрів серед всіх комбінацій")

# Додаткова інформація в sidebar
st.sidebar.markdown("---")
st.sidebar.header("ℹ️ Інформація")
st.sidebar.write("""
**Версія:** 1.0  
**Автор:** Crypto Trading Analyst  
**Оновлено:** 2024

**Можливості:**
- Оптимізація параметрів
- Симуляція торгівлі  
- Візуалізація результатів
- Експорт даних

**Підтримувані формати:**
- CSV з роздільником ;
- Хвилинні дані OHLCV
""")

st.sidebar.markdown("---")
st.sidebar.write("💡 **Порада:** Використовуйте оптимізацію для знаходження найкращих параметрів перед тестуванням на нових даних.")