import 'package:flutter/material.dart';
import 'package:flutter/services.dart';
import 'package:traffic_violation_app/theme/app_theme.dart';
import 'package:traffic_violation_app/models/violation.dart';
import 'package:intl/intl.dart';
import 'package:qr_flutter/qr_flutter.dart';

class PaymentScreen extends StatefulWidget {
  const PaymentScreen({super.key});

  @override
  State<PaymentScreen> createState() => _PaymentScreenState();
}

class _PaymentScreenState extends State<PaymentScreen> {
  String selectedMethod = 'bank_transfer';
  String selectedBank = 'vietcombank';
  
  final Map<String, Map<String, String>> bankInfo = {
    'vietcombank': {
      'name': 'Vietcombank',
      'accountNumber': '1234567890',
      'accountName': 'CỤC CSGT - BỘ CÔNG AN',
      'logo': 'https://api.vietqr.io/img/VCB.png',
    },
    'techcombank': {
      'name': 'Techcombank',
      'accountNumber': '9876543210',
      'accountName': 'CỤC CSGT - BỘ CÔNG AN',
      'logo': 'https://api.vietqr.io/img/TCB.png',
    },
    'vietinbank': {
      'name': 'VietinBank',
      'accountNumber': '5555666677',
      'accountName': 'CỤC CSGT - BỘ CÔNG AN',
      'logo': 'https://api.vietqr.io/img/CTG.png',
    },
  };

  @override
  Widget build(BuildContext context) {
    final violation = ModalRoute.of(context)!.settings.arguments as Violation;
    final currencyFormatter = NumberFormat.currency(locale: 'vi_VN', symbol: '₫');

    return Scaffold(
      appBar: AppBar(
        title: const Text('Thanh toán'),
      ),
      body: SingleChildScrollView(
        padding: const EdgeInsets.all(20),
        child: Column(
          crossAxisAlignment: CrossAxisAlignment.start,
          children: [
            // Amount Card
            Container(
              padding: const EdgeInsets.all(24),
              decoration: BoxDecoration(
                gradient: AppTheme.primaryGradient,
                borderRadius: BorderRadius.circular(20),
                boxShadow: [
                  BoxShadow(
                    color: AppTheme.primaryColor.withOpacity(0.3),
                    blurRadius: 20,
                    offset: const Offset(0, 10),
                  ),
                ],
              ),
              child: Column(
                children: [
                  const Text(
                    'Số tiền cần thanh toán',
                    style: TextStyle(
                      color: Colors.white70,
                      fontSize: 16,
                    ),
                  ),
                  const SizedBox(height: 12),
                  Text(
                    currencyFormatter.format(violation.fineAmount),
                    style: const TextStyle(
                      color: Colors.white,
                      fontSize: 40,
                      fontWeight: FontWeight.bold,
                    ),
                  ),
                  const SizedBox(height: 16),
                  Container(
                    padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 8),
                    decoration: BoxDecoration(
                      color: Colors.white.withOpacity(0.2),
                      borderRadius: BorderRadius.circular(20),
                    ),
                    child: Text(
                      'Vi phạm: ${violation.violationType}',
                      style: const TextStyle(
                        color: Colors.white,
                        fontSize: 14,
                      ),
                    ),
                  ),
                ],
              ),
            ),
            const SizedBox(height: 32),
            
            // Payment Methods
            const Text(
              'Phương thức thanh toán',
              style: TextStyle(
                fontSize: 20,
                fontWeight: FontWeight.bold,
              ),
            ),
            const SizedBox(height: 16),
            
            _buildPaymentMethod(
              'bank_transfer',
              'Chuyển khoản ngân hàng',
              Icons.account_balance,
              'Miễn phí',
            ),
            const SizedBox(height: 12),
            _buildPaymentMethod(
              'momo',
              'Ví MoMo',
              Icons.account_balance_wallet,
              'Miễn phí',
            ),
            const SizedBox(height: 12),
            _buildPaymentMethod(
              'vnpay',
              'VNPay',
              Icons.payment,
              'Miễn phí',
            ),
            const SizedBox(height: 32),
            
            // Payment Details
            if (selectedMethod == 'bank_transfer') ...[
              const Text(
                'Chọn ngân hàng',
                style: TextStyle(
                  fontSize: 20,
                  fontWeight: FontWeight.bold,
                ),
              ),
              const SizedBox(height: 16),
              
              ...bankInfo.entries.map((entry) => _buildBankOption(entry.key, entry.value)),
              
              const SizedBox(height: 24),
              
              // Bank Transfer Details
              Container(
                padding: const EdgeInsets.all(20),
                decoration: BoxDecoration(
                  color: Colors.white,
                  borderRadius: BorderRadius.circular(16),
                  border: Border.all(color: Colors.grey[300]!),
                ),
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    const Text(
                      'Thông tin chuyển khoản',
                      style: TextStyle(
                        fontSize: 18,
                        fontWeight: FontWeight.bold,
                      ),
                    ),
                    const SizedBox(height: 20),
                    
                    _buildTransferInfo(
                      'Ngân hàng',
                      bankInfo[selectedBank]!['name']!,
                      false,
                    ),
                    _buildTransferInfo(
                      'Số tài khoản',
                      bankInfo[selectedBank]!['accountNumber']!,
                      true,
                    ),
                    _buildTransferInfo(
                      'Tên tài khoản',
                      bankInfo[selectedBank]!['accountName']!,
                      false,
                    ),
                    _buildTransferInfo(
                      'Số tiền',
                      currencyFormatter.format(violation.fineAmount),
                      true,
                    ),
                    _buildTransferInfo(
                      'Nội dung',
                      'PHAT ${violation.licensePlate} ${violation.violationCode}',
                      true,
                    ),
                    
                    const SizedBox(height: 24),
                    
                    // QR Code
                    Center(
                      child: Column(
                        children: [
                          const Text(
                            'Quét mã QR để thanh toán',
                            style: TextStyle(
                              fontSize: 16,
                              fontWeight: FontWeight.w600,
                            ),
                          ),
                          const SizedBox(height: 16),
                          Container(
                            padding: const EdgeInsets.all(16),
                            decoration: BoxDecoration(
                              color: Colors.white,
                              borderRadius: BorderRadius.circular(16),
                              border: Border.all(color: AppTheme.primaryColor, width: 2),
                            ),
                            child: QrImageView(
                              data: _generateQRData(violation),
                              version: QrVersions.auto,
                              size: 200,
                            ),
                          ),
                          const SizedBox(height: 12),
                          Text(
                            'Hỗ trợ tất cả ứng dụng ngân hàng',
                            style: TextStyle(
                              fontSize: 12,
                              color: Colors.grey[600],
                            ),
                          ),
                        ],
                      ),
                    ),
                  ],
                ),
              ),
            ],
            
            if (selectedMethod == 'momo' || selectedMethod == 'vnpay') ...[
              Container(
                padding: const EdgeInsets.all(20),
                decoration: BoxDecoration(
                  color: Colors.white,
                  borderRadius: BorderRadius.circular(16),
                  border: Border.all(color: Colors.grey[300]!),
                ),
                child: Column(
                  children: [
                    Icon(
                      selectedMethod == 'momo'
                          ? Icons.account_balance_wallet
                          : Icons.payment,
                      size: 64,
                      color: AppTheme.primaryColor,
                    ),
                    const SizedBox(height: 16),
                    Text(
                      'Bạn sẽ được chuyển đến ứng dụng ${selectedMethod == 'momo' ? 'MoMo' : 'VNPay'}',
                      textAlign: TextAlign.center,
                      style: const TextStyle(fontSize: 16),
                    ),
                  ],
                ),
              ),
            ],
            
            const SizedBox(height: 32),
            
            // Warning
            Container(
              padding: const EdgeInsets.all(16),
              decoration: BoxDecoration(
                color: AppTheme.warningColor.withOpacity(0.1),
                borderRadius: BorderRadius.circular(12),
                border: Border.all(
                  color: AppTheme.warningColor.withOpacity(0.3),
                ),
              ),
              child: Row(
                children: [
                  const Icon(
                    Icons.info_outline,
                    color: AppTheme.warningColor,
                  ),
                  const SizedBox(width: 12),
                  Expanded(
                    child: Text(
                      'Vui lòng chuyển khoản đúng nội dung để hệ thống tự động xác nhận thanh toán',
                      style: TextStyle(
                        fontSize: 14,
                        color: Colors.grey[800],
                      ),
                    ),
                  ),
                ],
              ),
            ),
            const SizedBox(height: 100),
          ],
        ),
      ),
      bottomNavigationBar: Container(
        padding: const EdgeInsets.all(20),
        decoration: BoxDecoration(
          color: Theme.of(context).scaffoldBackgroundColor,
          boxShadow: [
            BoxShadow(
              color: Colors.black.withOpacity(0.1),
              blurRadius: 10,
              offset: const Offset(0, -5),
            ),
          ],
        ),
        child: SizedBox(
          width: double.infinity,
          height: 56,
          child: Container(
            decoration: BoxDecoration(
              gradient: AppTheme.primaryGradient,
              borderRadius: BorderRadius.circular(12),
              boxShadow: [
                BoxShadow(
                  color: AppTheme.primaryColor.withOpacity(0.3),
                  blurRadius: 12,
                  offset: const Offset(0, 6),
                ),
              ],
            ),
            child: ElevatedButton(
              onPressed: () {
                _showPaymentConfirmation(context, violation);
              },
              style: ElevatedButton.styleFrom(
                backgroundColor: Colors.transparent,
                shadowColor: Colors.transparent,
                shape: RoundedRectangleBorder(
                  borderRadius: BorderRadius.circular(12),
                ),
              ),
              child: const Text(
                'Xác nhận thanh toán',
                style: TextStyle(
                  fontSize: 16,
                  fontWeight: FontWeight.w600,
                  color: Colors.white,
                ),
              ),
            ),
          ),
        ),
      ),
    );
  }

  Widget _buildPaymentMethod(String value, String title, IconData icon, String subtitle) {
    final isSelected = selectedMethod == value;
    
    return Container(
      decoration: BoxDecoration(
        color: Colors.white,
        borderRadius: BorderRadius.circular(12),
        border: Border.all(
          color: isSelected ? AppTheme.primaryColor : Colors.grey[300]!,
          width: isSelected ? 2 : 1,
        ),
      ),
      child: RadioListTile(
        value: value,
        groupValue: selectedMethod,
        onChanged: (v) => setState(() => selectedMethod = v.toString()),
        title: Text(
          title,
          style: TextStyle(
            fontWeight: isSelected ? FontWeight.w600 : FontWeight.normal,
          ),
        ),
        subtitle: Text(subtitle),
        secondary: Icon(
          icon,
          color: isSelected ? AppTheme.primaryColor : Colors.grey,
        ),
        activeColor: AppTheme.primaryColor,
      ),
    );
  }

  Widget _buildBankOption(String value, Map<String, String> info) {
    final isSelected = selectedBank == value;
    
    return Container(
      margin: const EdgeInsets.only(bottom: 12),
      decoration: BoxDecoration(
        color: Colors.white,
        borderRadius: BorderRadius.circular(12),
        border: Border.all(
          color: isSelected ? AppTheme.primaryColor : Colors.grey[300]!,
          width: isSelected ? 2 : 1,
        ),
      ),
      child: RadioListTile(
        value: value,
        groupValue: selectedBank,
        onChanged: (v) => setState(() => selectedBank = v.toString()),
        title: Text(
          info['name']!,
          style: TextStyle(
            fontWeight: isSelected ? FontWeight.w600 : FontWeight.normal,
          ),
        ),
        subtitle: Text(info['accountNumber']!),
        secondary: ClipRRect(
          borderRadius: BorderRadius.circular(8),
          child: Image.network(
            info['logo']!,
            width: 40,
            height: 40,
            fit: BoxFit.cover,
          ),
        ),
        activeColor: AppTheme.primaryColor,
      ),
    );
  }

  Widget _buildTransferInfo(String label, String value, bool canCopy) {
    return Padding(
      padding: const EdgeInsets.only(bottom: 16),
      child: Row(
        mainAxisAlignment: MainAxisAlignment.spaceBetween,
        children: [
          Text(
            label,
            style: TextStyle(
              fontSize: 14,
              color: Colors.grey[600],
            ),
          ),
          Row(
            children: [
              Text(
                value,
                style: const TextStyle(
                  fontSize: 14,
                  fontWeight: FontWeight.w600,
                ),
              ),
              if (canCopy) ...[
                const SizedBox(width: 8),
                InkWell(
                  onTap: () {
                    Clipboard.setData(ClipboardData(text: value));
                    ScaffoldMessenger.of(context).showSnackBar(
                      const SnackBar(content: Text('Đã sao chép')),
                    );
                  },
                  child: Icon(
                    Icons.copy,
                    size: 16,
                    color: AppTheme.primaryColor,
                  ),
                ),
              ],
            ],
          ),
        ],
      ),
    );
  }

  String _generateQRData(Violation violation) {
    final bank = bankInfo[selectedBank]!;
    return 'bank=${bank['name']}&account=${bank['accountNumber']}&amount=${violation.fineAmount.toInt()}&content=PHAT ${violation.licensePlate} ${violation.violationCode}';
  }

  void _showPaymentConfirmation(BuildContext context, Violation violation) {
    showDialog(
      context: context,
      builder: (context) => AlertDialog(
        shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(20)),
        title: const Text('Xác nhận thanh toán'),
        content: const Text(
          'Bạn đã hoàn tất chuyển khoản?\n\nHệ thống sẽ xác nhận giao dịch trong vòng 5-10 phút.',
        ),
        actions: [
          TextButton(
            onPressed: () => Navigator.pop(context),
            child: const Text('Hủy'),
          ),
          ElevatedButton(
            onPressed: () {
              Navigator.pop(context);
              _showSuccessDialog(context);
            },
            style: ElevatedButton.styleFrom(
              backgroundColor: AppTheme.primaryColor,
              foregroundColor: Colors.white,
            ),
            child: const Text('Đã chuyển khoản'),
          ),
        ],
      ),
    );
  }

  void _showSuccessDialog(BuildContext context) {
    showDialog(
      context: context,
      barrierDismissible: false,
      builder: (context) => AlertDialog(
        shape: RoundedRectangleBorder(borderRadius: BorderRadius.circular(20)),
        content: Column(
          mainAxisSize: MainAxisSize.min,
          children: [
            Container(
              width: 80,
              height: 80,
              decoration: BoxDecoration(
                color: AppTheme.successColor.withOpacity(0.2),
                shape: BoxShape.circle,
              ),
              child: const Icon(
                Icons.check_circle,
                color: AppTheme.successColor,
                size: 50,
              ),
            ),
            const SizedBox(height: 24),
            const Text(
              'Đã ghi nhận thanh toán',
              style: TextStyle(
                fontSize: 20,
                fontWeight: FontWeight.bold,
              ),
            ),
            const SizedBox(height: 12),
            Text(
              'Chúng tôi đã nhận được yêu cầu thanh toán của bạn. Hệ thống sẽ xác nhận trong vòng 5-10 phút.',
              textAlign: TextAlign.center,
              style: TextStyle(
                fontSize: 14,
                color: Colors.grey[600],
              ),
            ),
          ],
        ),
        actions: [
          SizedBox(
            width: double.infinity,
            child: ElevatedButton(
              onPressed: () {
                Navigator.pop(context);
                Navigator.pop(context);
                Navigator.pop(context);
              },
              style: ElevatedButton.styleFrom(
                backgroundColor: AppTheme.primaryColor,
                foregroundColor: Colors.white,
              ),
              child: const Text('Về trang chủ'),
            ),
          ),
        ],
      ),
    );
  }
}
