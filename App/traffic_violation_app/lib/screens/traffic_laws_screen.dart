import 'package:flutter/material.dart';
import 'package:traffic_violation_app/theme/app_theme.dart';
import 'package:traffic_violation_app/data/mock_data.dart';
import 'package:intl/intl.dart';
import 'package:traffic_violation_app/services/app_settings.dart';

class TrafficLawsScreen extends StatefulWidget {
  const TrafficLawsScreen({super.key});

  @override
  State<TrafficLawsScreen> createState() => _TrafficLawsScreenState();
}

class _TrafficLawsScreenState extends State<TrafficLawsScreen> {
  final TextEditingController _searchController = TextEditingController();
  final AppSettings _s = AppSettings();
  String _searchQuery = '';
  String _selectedCategory = 'Tất cả';

  // Vietnamese category names (used as keys for filtering MockData)
  final List<String> _categoryKeys = [
    'Tất cả',
    'Đèn đỏ',
    'Tốc độ',
    'Mũ bảo hiểm',
    'Làn đường',
    'Nồng độ cồn',
    'Dừng đỗ',
    'Biển số xe',
    'Điện thoại',
  ];

  // English translations for category names
  final Map<String, String> _categoryTranslations = {
    'Tất cả': 'All',
    'Đèn đỏ': 'Red light',
    'Tốc độ': 'Speeding',
    'Mũ bảo hiểm': 'Helmet',
    'Làn đường': 'Lane',
    'Nồng độ cồn': 'Alcohol',
    'Dừng đỗ': 'Parking',
    'Biển số xe': 'License plate',
    'Điện thoại': 'Phone',
  };

  @override
  void initState() {
    super.initState();
    _s.addListener(_onSettingsChanged);
  }

  @override
  void dispose() {
    _searchController.dispose();
    _s.removeListener(_onSettingsChanged);
    super.dispose();
  }

  void _onSettingsChanged() {
    if (mounted) setState(() {});
  }

  String _getCategoryLabel(String key) {
    if (_s.isVietnamese) return key;
    return _categoryTranslations[key] ?? key;
  }

  @override
  Widget build(BuildContext context) {
    final filteredLaws = _getFilteredLaws();
    final currencyFormatter = NumberFormat.currency(locale: 'vi_VN', symbol: '₫');
    final isDark = Theme.of(context).brightness == Brightness.dark;
    final bgColor = isDark ? const Color(0xFF121212) : AppTheme.surfaceColor;
    final cardBg = isDark ? const Color(0xFF1E1E1E) : Colors.white;
    final textPrimary = isDark ? const Color(0xFFE0E0E0) : AppTheme.textPrimary;
    final textSecondary = isDark ? const Color(0xFF9E9E9E) : AppTheme.textSecondary;

    return Scaffold(
      backgroundColor: bgColor,
      appBar: AppBar(
        title: Text(_s.tr('Tra cứu luật giao thông', 'Traffic Law Lookup')),
        backgroundColor: isDark ? const Color(0xFF1E1E1E) : Colors.white,
        foregroundColor: textPrimary,
        elevation: 0,
        scrolledUnderElevation: 1,
      ),
      body: Column(
        children: [
          // Search Bar
          Container(
            padding: const EdgeInsets.fromLTRB(16, 8, 16, 12),
            decoration: BoxDecoration(
              color: isDark ? const Color(0xFF1E1E1E) : Colors.white,
              boxShadow: [
                BoxShadow(
                  color: Colors.black.withValues(alpha: 0.05),
                  blurRadius: 10,
                  offset: const Offset(0, 2),
                ),
              ],
            ),
            child: TextField(
              controller: _searchController,
              style: TextStyle(color: textPrimary),
              decoration: InputDecoration(
                hintText: _s.tr('Tìm kiếm luật, mức phạt...', 'Search laws, fines...'),
                hintStyle: TextStyle(color: textSecondary),
                prefixIcon: Icon(Icons.search, color: textSecondary),
                suffixIcon: _searchQuery.isNotEmpty
                    ? IconButton(
                        icon: Icon(Icons.clear, color: textSecondary),
                        onPressed: () {
                          _searchController.clear();
                          setState(() => _searchQuery = '');
                        },
                      )
                    : null,
                filled: true,
                fillColor: isDark ? const Color(0xFF2A2A2A) : AppTheme.surfaceColor,
                border: OutlineInputBorder(
                  borderRadius: BorderRadius.circular(14),
                  borderSide: BorderSide.none,
                ),
              ),
              onChanged: (value) {
                setState(() => _searchQuery = value);
              },
            ),
          ),

          // Categories
          SizedBox(
            height: 52,
            child: ListView.builder(
              scrollDirection: Axis.horizontal,
              padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 8),
              itemCount: _categoryKeys.length,
              itemBuilder: (context, index) {
                final categoryKey = _categoryKeys[index];
                final isSelected = _selectedCategory == categoryKey;
                final label = _getCategoryLabel(categoryKey);

                return Padding(
                  padding: const EdgeInsets.only(right: 8),
                  child: GestureDetector(
                    onTap: () => setState(() => _selectedCategory = categoryKey),
                    child: AnimatedContainer(
                      duration: const Duration(milliseconds: 250),
                      padding: const EdgeInsets.symmetric(horizontal: 16, vertical: 8),
                      decoration: BoxDecoration(
                        color: isSelected ? AppTheme.primaryColor : (isDark ? const Color(0xFF2A2A2A) : Colors.white),
                        borderRadius: BorderRadius.circular(20),
                        border: Border.all(
                          color: isSelected ? AppTheme.primaryColor : (isDark ? const Color(0xFF3A3A3A) : AppTheme.dividerColor),
                        ),
                        boxShadow: isSelected
                            ? [BoxShadow(color: AppTheme.primaryColor.withValues(alpha: 0.3), blurRadius: 8, offset: const Offset(0, 2))]
                            : null,
                      ),
                      child: Text(
                        label,
                        style: TextStyle(
                          color: isSelected ? Colors.white : textSecondary,
                          fontWeight: isSelected ? FontWeight.w600 : FontWeight.normal,
                          fontSize: 13,
                        ),
                      ),
                    ),
                  ),
                );
              },
            ),
          ),

          // Laws List
          Expanded(
            child: filteredLaws.isEmpty
                ? Center(
                    child: Column(
                      mainAxisAlignment: MainAxisAlignment.center,
                      children: [
                        Container(
                          width: 80,
                          height: 80,
                          decoration: BoxDecoration(
                            color: textSecondary.withValues(alpha: 0.08),
                            shape: BoxShape.circle,
                          ),
                          child: Icon(
                            Icons.search_off,
                            size: 40,
                            color: textSecondary.withValues(alpha: 0.4),
                          ),
                        ),
                        const SizedBox(height: 16),
                        Text(
                          _s.tr('Không tìm thấy kết quả', 'No results found'),
                          style: TextStyle(
                            fontSize: 16,
                            fontWeight: FontWeight.w600,
                            color: textSecondary,
                          ),
                        ),
                        const SizedBox(height: 6),
                        Text(
                          _s.tr('Thử tìm kiếm với từ khóa khác', 'Try a different keyword'),
                          style: TextStyle(
                            fontSize: 13,
                            color: textSecondary.withValues(alpha: 0.6),
                          ),
                        ),
                      ],
                    ),
                  )
                : ListView.builder(
                    padding: const EdgeInsets.fromLTRB(16, 8, 16, 80),
                    itemCount: filteredLaws.length,
                    itemBuilder: (context, index) {
                      final law = filteredLaws[index];
                      return TweenAnimationBuilder<double>(
                        tween: Tween(begin: 0, end: 1),
                        duration: Duration(milliseconds: 350 + index * 60),
                        curve: Curves.easeOutCubic,
                        builder: (context, anim, child) {
                          return Opacity(
                            opacity: anim,
                            child: Transform.translate(
                              offset: Offset(0, 15 * (1 - anim)),
                              child: child,
                            ),
                          );
                        },
                        child: _buildLawCard(law, currencyFormatter, isDark, cardBg, textPrimary, textSecondary),
                      );
                    },
                  ),
          ),
        ],
      ),
    );
  }

  List _getFilteredLaws() {
    var laws = MockData.trafficLaws;

    // Filter by category
    if (_selectedCategory != 'Tất cả') {
      laws = laws.where((law) => law.category == _selectedCategory).toList();
    }

    // Filter by search query
    if (_searchQuery.isNotEmpty) {
      laws = laws.where((law) {
        final query = _searchQuery.toLowerCase();
        return law.title.toLowerCase().contains(query) ||
               law.description.toLowerCase().contains(query) ||
               law.code.toLowerCase().contains(query);
      }).toList();
    }

    return laws;
  }

  Widget _buildLawCard(law, currencyFormatter, bool isDark, Color cardBg, Color textPrimary, Color textSecondary) {
    return Container(
      margin: const EdgeInsets.only(bottom: 14),
      decoration: BoxDecoration(
        color: cardBg,
        borderRadius: BorderRadius.circular(16),
        border: Border.all(
          color: isDark ? const Color(0xFF333333) : AppTheme.dividerColor,
          width: 0.5,
        ),
        boxShadow: [
          BoxShadow(
            color: Colors.black.withValues(alpha: isDark ? 0.15 : 0.05),
            blurRadius: 10,
            offset: const Offset(0, 2),
          ),
        ],
      ),
      child: Material(
        color: Colors.transparent,
        child: InkWell(
          borderRadius: BorderRadius.circular(16),
          onTap: () {
            _showLawDetail(law, currencyFormatter, isDark, cardBg, textPrimary, textSecondary);
          },
          child: Padding(
            padding: const EdgeInsets.all(18),
            child: Column(
              crossAxisAlignment: CrossAxisAlignment.start,
              children: [
                Row(
                  children: [
                    Container(
                      padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 6),
                      decoration: BoxDecoration(
                        color: AppTheme.primaryColor.withValues(alpha: 0.1),
                        borderRadius: BorderRadius.circular(8),
                      ),
                      child: Text(
                        law.code,
                        style: const TextStyle(
                          color: AppTheme.primaryColor,
                          fontSize: 12,
                          fontWeight: FontWeight.w600,
                        ),
                      ),
                    ),
                    const SizedBox(width: 8),
                    Container(
                      padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 6),
                      decoration: BoxDecoration(
                        color: isDark ? const Color(0xFF2A2A2A) : Colors.grey[200],
                        borderRadius: BorderRadius.circular(8),
                      ),
                      child: Text(
                        _s.isVietnamese ? law.category : (_categoryTranslations[law.category] ?? law.category),
                        style: TextStyle(
                          color: textSecondary,
                          fontSize: 12,
                          fontWeight: FontWeight.w500,
                        ),
                      ),
                    ),
                  ],
                ),
                const SizedBox(height: 12),

                Text(
                  law.title,
                  style: TextStyle(
                    fontSize: 15,
                    fontWeight: FontWeight.bold,
                    color: textPrimary,
                  ),
                ),
                const SizedBox(height: 8),

                Text(
                  law.description,
                  style: TextStyle(
                    fontSize: 13,
                    color: textSecondary,
                    height: 1.4,
                  ),
                  maxLines: 2,
                  overflow: TextOverflow.ellipsis,
                ),
                const SizedBox(height: 12),

                Wrap(
                  spacing: 8,
                  runSpacing: 8,
                  children: law.fineLevels.map<Widget>((fineLevel) {
                    return Container(
                      padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 6),
                      decoration: BoxDecoration(
                        color: AppTheme.dangerColor.withValues(alpha: 0.08),
                        borderRadius: BorderRadius.circular(8),
                      ),
                      child: Text(
                        '${fineLevel.vehicleType}: ${currencyFormatter.format(fineLevel.minAmount)} - ${currencyFormatter.format(fineLevel.maxAmount)}',
                        style: const TextStyle(
                          color: AppTheme.dangerColor,
                          fontSize: 12,
                          fontWeight: FontWeight.w600,
                        ),
                      ),
                    );
                  }).toList(),
                ),
              ],
            ),
          ),
        ),
      ),
    );
  }

  void _showLawDetail(law, currencyFormatter, bool isDark, Color cardBg, Color textPrimary, Color textSecondary) {
    showModalBottomSheet(
      context: context,
      isScrollControlled: true,
      backgroundColor: Colors.transparent,
      builder: (context) => Container(
        height: MediaQuery.of(context).size.height * 0.8,
        decoration: BoxDecoration(
          color: cardBg,
          borderRadius: const BorderRadius.vertical(top: Radius.circular(28)),
        ),
        child: Column(
          children: [
            // Handle
            Container(
              margin: const EdgeInsets.symmetric(vertical: 12),
              width: 40,
              height: 4,
              decoration: BoxDecoration(
                color: isDark ? const Color(0xFF555555) : Colors.grey[300],
                borderRadius: BorderRadius.circular(2),
              ),
            ),

            Expanded(
              child: SingleChildScrollView(
                padding: const EdgeInsets.all(24),
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.start,
                  children: [
                    Container(
                      padding: const EdgeInsets.symmetric(horizontal: 12, vertical: 6),
                      decoration: BoxDecoration(
                        color: AppTheme.primaryColor.withValues(alpha: 0.1),
                        borderRadius: BorderRadius.circular(8),
                      ),
                      child: Text(
                        law.code,
                        style: const TextStyle(
                          color: AppTheme.primaryColor,
                          fontSize: 14,
                          fontWeight: FontWeight.w600,
                        ),
                      ),
                    ),
                    const SizedBox(height: 16),

                    Text(
                      law.title,
                      style: TextStyle(
                        fontSize: 22,
                        fontWeight: FontWeight.bold,
                        color: textPrimary,
                      ),
                    ),
                    const SizedBox(height: 16),

                    Text(
                      law.description,
                      style: TextStyle(
                        fontSize: 15,
                        color: textSecondary,
                        height: 1.6,
                      ),
                    ),
                    const SizedBox(height: 24),

                    Text(
                      _s.tr('Mức phạt', 'Fine levels'),
                      style: TextStyle(
                        fontSize: 18,
                        fontWeight: FontWeight.bold,
                        color: textPrimary,
                      ),
                    ),
                    const SizedBox(height: 12),

                    ...law.fineLevels.map((fineLevel) {
                      return Container(
                        margin: const EdgeInsets.only(bottom: 12),
                        padding: const EdgeInsets.all(16),
                        decoration: BoxDecoration(
                          color: AppTheme.dangerColor.withValues(alpha: 0.05),
                          borderRadius: BorderRadius.circular(14),
                          border: Border.all(
                            color: AppTheme.dangerColor.withValues(alpha: 0.15),
                          ),
                        ),
                        child: Column(
                          crossAxisAlignment: CrossAxisAlignment.start,
                          children: [
                            Text(
                              fineLevel.vehicleType,
                              style: TextStyle(
                                fontSize: 15,
                                fontWeight: FontWeight.w600,
                                color: textPrimary,
                              ),
                            ),
                            const SizedBox(height: 8),
                            Text(
                              '${currencyFormatter.format(fineLevel.minAmount)} - ${currencyFormatter.format(fineLevel.maxAmount)}',
                              style: const TextStyle(
                                fontSize: 18,
                                fontWeight: FontWeight.bold,
                                color: AppTheme.dangerColor,
                              ),
                            ),
                            if (fineLevel.additionalPenalty != null) ...[
                              const SizedBox(height: 8),
                              Text(
                                fineLevel.additionalPenalty!,
                                style: TextStyle(
                                  fontSize: 13,
                                  color: textSecondary,
                                ),
                              ),
                            ],
                          ],
                        ),
                      );
                    }),

                    const SizedBox(height: 24),

                    Container(
                      padding: const EdgeInsets.all(16),
                      decoration: BoxDecoration(
                        color: isDark ? const Color(0xFF2A2A2A) : Colors.grey[100],
                        borderRadius: BorderRadius.circular(14),
                      ),
                      child: Column(
                        crossAxisAlignment: CrossAxisAlignment.start,
                        children: [
                          Text(
                            _s.tr('Căn cứ pháp lý', 'Legal basis'),
                            style: TextStyle(
                              fontSize: 13,
                              color: textSecondary,
                            ),
                          ),
                          const SizedBox(height: 4),
                          Text(
                            law.lawReference,
                            style: TextStyle(
                              fontSize: 15,
                              fontWeight: FontWeight.w600,
                              color: textPrimary,
                            ),
                          ),
                        ],
                      ),
                    ),
                  ],
                ),
              ),
            ),
          ],
        ),
      ),
    );
  }
}
