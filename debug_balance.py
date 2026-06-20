import os
from pybit.unified_trading import HTTP

session = HTTP(
    testnet=True,
    api_key=os.environ["BYBIT_API_KEY"],
    api_secret=os.environ["BYBIT_API_SECRET"],
)

print("=== FUND ===")
print(session.get_coins_balance(accountType="FUND"))

print("=== CONTRACT ===")
try:
    print(session.get_wallet_balance(accountType="CONTRACT"))
except Exception as e:
    print("CONTRACT error:", e)

print("=== SPOT ===")
try:
    print(session.get_coins_balance(accountType="SPOT"))
except Exception as e:
    print("SPOT error:", e)
