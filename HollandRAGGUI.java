import javax.swing.*;
import javax.swing.border.*;
import javax.swing.filechooser.FileNameExtensionFilter;
import java.awt.*;
import java.awt.event.*;
import java.awt.geom.RoundRectangle2D;
import java.io.*;
import java.net.URI;
import java.nio.charset.StandardCharsets;
import java.util.ArrayList;
import java.util.List;
import javax.swing.event.HyperlinkEvent;
import javax.swing.event.HyperlinkListener;
import java.util.regex.Matcher;
import java.util.regex.Pattern;

/**
 * HollandRAGGUI - Một giao diện Java Swing hiện đại cho hệ thống Agentic RAG.
 * Phiên bản REDESIGN: Bo tròn 100%, Phân loại màu sắc theo thương hiệu GPA, Sửa lỗi tự động gửi tin.
 */
public class HollandRAGGUI extends JFrame {

    private final Color COLOR_BG = new Color(255, 255, 255);
    private final Color COLOR_SIDEBAR = new Color(248, 249, 250); 
    private final Color COLOR_GPA_BLUE = new Color(26, 75, 158); 
    private final Color COLOR_ACCENT = new Color(255, 193, 7);
    private final Color COLOR_BORDER = new Color(230, 233, 237);
    private final Color COLOR_TEXT_DARK = new Color(33, 37, 41);
    private final Color COLOR_AI_BUBBLE = new Color(242, 243, 245);
    private final Color COLOR_USER_BUBBLE = COLOR_GPA_BLUE;

    private JPanel chatHistoryPanel;
    private JPanel messagesWrapper; 
    private JScrollPane scrollPane;
    private JTextField inputField;
    private JButton sendButton;
    private JButton uploadButton;
    private String selectedFilePath = null;
    private JLabel statusLabel;
    private JLabel agentLabel;
    private JLabel charCounterLabel;
    private JPanel chipsPanel;
    private boolean isReady = false;

    private static final String PYTHON_EXECUTABLE = ".venv312/bin/python";
    private static final String BRIDGE_SCRIPT = "java_bridge.py";

    public HollandRAGGUI() {
        setTitle("GPA Agentic RAG Advisor");
        setSize(1100, 800);
        setDefaultCloseOperation(JFrame.EXIT_ON_CLOSE);
        setLocationRelativeTo(null);
        getContentPane().setBackground(COLOR_BG);
        initUI();
        isReady = true; 
    }

    private void initUI() {
        setLayout(new BorderLayout());

        // --- 1. HEADER ---
        JPanel header = new JPanel(new BorderLayout());
        header.setBackground(COLOR_BG);
        header.setPreferredSize(new Dimension(0, 70));
        header.setBorder(new MatteBorder(0, 0, 1, 0, COLOR_BORDER));
        JPanel branding = new JPanel(new FlowLayout(FlowLayout.CENTER, 15, 12));
        branding.setOpaque(false);
        try {
            ImageIcon icon = new ImageIcon("logo.png");
            Image img = icon.getImage().getScaledInstance(45, 45, Image.SCALE_SMOOTH);
            branding.add(new JLabel(new ImageIcon(img)));
        } catch (Exception e) {}
        JLabel titleLabel = new JLabel("Trợ lý Hướng nghiệp GPA AI");
        titleLabel.setFont(new Font("SansSerif", Font.BOLD, 18));
        branding.add(titleLabel);
        header.add(branding, BorderLayout.CENTER);
        add(header, BorderLayout.NORTH);

        // --- 2. SIDEBAR ---
        JPanel sidebar = new JPanel(new BorderLayout());
        sidebar.setPreferredSize(new Dimension(260, 0));
        sidebar.setBackground(COLOR_SIDEBAR);
        sidebar.setBorder(new MatteBorder(0, 0, 0, 1, COLOR_BORDER));
        JPanel sideContent = new JPanel();
        sideContent.setLayout(new BoxLayout(sideContent, BoxLayout.Y_AXIS));
        sideContent.setOpaque(false);
        sideContent.setBorder(new EmptyBorder(20, 15, 20, 15));
        
        // Nút chính: Màu xanh GPA
        JButton newChatBtn = createSidebarButton("Trò chuyện mới", true);
        newChatBtn.addActionListener(e -> resetChat());
        sideContent.add(newChatBtn);
        sideContent.add(Box.createRigidArea(new Dimension(0, 30))); // Khoảng cách lớn hơn

        // Nhãn Gợi ý nhanh
        JLabel suggestLabel = new JLabel("GỢI Ý NHANH");
        suggestLabel.setFont(new Font("SansSerif", Font.BOLD, 11));
        suggestLabel.setForeground(new Color(150, 150, 150));
        sideContent.add(suggestLabel);
        sideContent.add(Box.createRigidArea(new Dimension(0, 10)));

        String[] suggestions = {"Tìm hiểu nhóm Holland", "Xem lộ trình ngoại khóa", "Tham khảo hồ sơ mẫu"};
        for (String s : suggestions) {
            JButton chip = createSidebarButton(s, false);
            chip.addActionListener(e -> {
                if (!isReady) return;
                if (s.equals("Tham khảo hồ sơ mẫu") && selectedFilePath == null) {
                    JOptionPane.showMessageDialog(this, 
                        "Vui lòng đính kèm CV (📎) để đối chiếu hồ sơ!", 
                        "Yêu cầu CV", JOptionPane.WARNING_MESSAGE);
                    selectFile();
                    if (selectedFilePath == null) return;
                }
                inputField.setText(s);
                sendMessage(s.equals("Tham khảo hồ sơ mẫu") ? "match_hoso" : null);
            });
            sideContent.add(chip);
            sideContent.add(Box.createRigidArea(new Dimension(0, 8))); // Khoảng cách giữa các chip đều hơn
        }

        sideContent.add(Box.createRigidArea(new Dimension(0, 25))); // Khoảng cách đến lịch sử
        JLabel historyLabel = new JLabel("LỊCH SỬ HỘI THOẠI");
        historyLabel.setFont(new Font("SansSerif", Font.BOLD, 11));
        historyLabel.setForeground(new Color(150, 150, 150));
        sideContent.add(historyLabel);
        sideContent.add(Box.createRigidArea(new Dimension(0, 10)));
        
        // Nút lịch sử: Màu trắng viền vàng
        JButton oldChatBtn = createSidebarButton("Cuộc hội thoại cũ", false);
        oldChatBtn.addActionListener(e -> resetChat());
        sideContent.add(oldChatBtn);
        
        sidebar.add(sideContent, BorderLayout.NORTH);
        JLabel footerLabel = new JLabel("© Trợ lý AI GPA 2026");
        footerLabel.setFont(new Font("SansSerif", Font.PLAIN, 11));
        footerLabel.setForeground(new Color(170, 170, 170));
        footerLabel.setBorder(new EmptyBorder(0, 20, 20, 0));
        sidebar.add(footerLabel, BorderLayout.SOUTH);
        add(sidebar, BorderLayout.WEST);

        // --- 3. MAIN CONTENT ---
        JPanel mainContent = new JPanel(new BorderLayout());
        mainContent.setBackground(COLOR_BG);

        chatHistoryPanel = new JPanel(new BorderLayout());
        chatHistoryPanel.setBackground(COLOR_BG);
        messagesWrapper = new JPanel();
        messagesWrapper.setLayout(new BoxLayout(messagesWrapper, BoxLayout.Y_AXIS));
        messagesWrapper.setBackground(COLOR_BG);
        messagesWrapper.setBorder(new EmptyBorder(20, 0, 20, 0));
        chatHistoryPanel.add(messagesWrapper, BorderLayout.NORTH);
        scrollPane = new JScrollPane(chatHistoryPanel);
        scrollPane.setBorder(null);
        scrollPane.getVerticalScrollBar().setUnitIncrement(16);
        mainContent.add(scrollPane, BorderLayout.CENTER);

        // Thanh Nhập liệu
        JPanel bottomContainer = new JPanel(new BorderLayout());
        bottomContainer.setBackground(COLOR_BG);
        bottomContainer.setBorder(new EmptyBorder(10, 0, 30, 0));
        JPanel pillPanel = new JPanel(new BorderLayout(10, 0)) {
            @Override protected void paintComponent(Graphics g) {
                Graphics2D g2 = (Graphics2D) g.create();
                g2.setRenderingHint(RenderingHints.KEY_ANTIALIASING, RenderingHints.VALUE_ANTIALIAS_ON);
                g2.setColor(Color.WHITE);
                g2.fill(new RoundRectangle2D.Double(0, 0, getWidth()-1, getHeight()-1, 50, 50));
                g2.setColor(new Color(210, 215, 220));
                g2.draw(new RoundRectangle2D.Double(0, 0, getWidth()-1, getHeight()-1, 50, 50));
                g2.dispose();
            }
        };
        pillPanel.setPreferredSize(new Dimension(720, 55));
        pillPanel.setOpaque(false);
        pillPanel.setBorder(new EmptyBorder(5, 20, 5, 20));
        uploadButton = new JButton("📎");
        uploadButton.setFont(new Font("SansSerif", Font.PLAIN, 20));
        uploadButton.setBorder(null);
        uploadButton.setContentAreaFilled(false);
        uploadButton.setFocusable(false);
        uploadButton.setCursor(new Cursor(Cursor.HAND_CURSOR));
        
        inputField = new JTextField();
        inputField.setBorder(null);
        inputField.setFont(new Font("SansSerif", Font.PLAIN, 15));
        inputField.setBackground(new Color(0,0,0,0));
        
        JPanel rightTools = new JPanel(new FlowLayout(FlowLayout.RIGHT, 10, 12));
        rightTools.setOpaque(false);
        charCounterLabel = new JLabel("0/500");
        charCounterLabel.setFont(new Font("SansSerif", Font.PLAIN, 12));
        charCounterLabel.setForeground(new Color(180, 180, 190));
        sendButton = new JButton("➤");
        sendButton.setFont(new Font("SansSerif", Font.BOLD, 20));
        sendButton.setBorder(null);
        sendButton.setContentAreaFilled(false);
        sendButton.setForeground(COLOR_GPA_BLUE);
        sendButton.setFocusable(false);
        sendButton.setCursor(new Cursor(Cursor.HAND_CURSOR));
        rightTools.add(charCounterLabel);
        rightTools.add(sendButton);
        pillPanel.add(uploadButton, BorderLayout.WEST);
        pillPanel.add(inputField, BorderLayout.CENTER);
        pillPanel.add(rightTools, BorderLayout.EAST);
        
        JPanel outerPill = new JPanel(new FlowLayout(FlowLayout.CENTER));
        outerPill.setOpaque(false);
        outerPill.add(pillPanel);
        
        JPanel metaPanel = new JPanel(new FlowLayout(FlowLayout.CENTER));
        metaPanel.setOpaque(false);
        agentLabel = new JLabel("Mode: Waiting...");
        agentLabel.setFont(new Font("SansSerif", Font.PLAIN, 12));
        statusLabel = new JLabel("● System Ready");
        statusLabel.setFont(new Font("SansSerif", Font.BOLD, 12));
        statusLabel.setForeground(new Color(46, 204, 113));
        metaPanel.add(agentLabel);
        metaPanel.add(new JLabel(" | "));
        metaPanel.add(statusLabel);
        
        JPanel southStack = new JPanel();
        southStack.setLayout(new BoxLayout(southStack, BoxLayout.Y_AXIS));
        southStack.setOpaque(true);
        southStack.setBackground(COLOR_BG);
        southStack.add(outerPill);
        southStack.add(metaPanel);
        bottomContainer.add(southStack, BorderLayout.CENTER);
        mainContent.add(bottomContainer, BorderLayout.SOUTH);

        add(mainContent, BorderLayout.CENTER);

        // Events
        sendButton.addActionListener(e -> sendMessage());
        inputField.addActionListener(e -> sendMessage());
        uploadButton.addActionListener(e -> selectFile());
        inputField.addKeyListener(new KeyAdapter() { public void keyReleased(KeyEvent e) { charCounterLabel.setText(inputField.getText().length() + "/500"); } });

        addChatMessage("Xin chào! Tôi là Trợ lý AI chuyên nghiệp của GPA. Hãy gửi CV hoặc đặt câu hỏi về hướng nghiệp cho tôi nhé!", false);
    }

    private JButton createSidebarButton(String text, boolean isPrimary) {
        JButton btn = new JButton(text) {
            @Override protected void paintComponent(Graphics g) {
                Graphics2D g2 = (Graphics2D) g.create();
                g2.setRenderingHint(RenderingHints.KEY_ANTIALIASING, RenderingHints.VALUE_ANTIALIAS_ON);
                g2.setColor(isPrimary ? COLOR_GPA_BLUE : Color.WHITE);
                g2.fill(new RoundRectangle2D.Double(0, 0, getWidth()-1, getHeight()-1, 20, 20));
                g2.setColor(isPrimary ? COLOR_GPA_BLUE : COLOR_ACCENT);
                g2.draw(new RoundRectangle2D.Double(0, 0, getWidth()-1, getHeight()-1, 20, 20));
                super.paintComponent(g2);
                g2.dispose();
            }
        };
        btn.setAlignmentX(Component.LEFT_ALIGNMENT);
        btn.setMaximumSize(new Dimension(4000, 45));
        btn.setFont(new Font("SansSerif", isPrimary ? Font.BOLD : Font.PLAIN, 14));
        btn.setForeground(isPrimary ? Color.WHITE : COLOR_TEXT_DARK);
        btn.setContentAreaFilled(false);
        btn.setFocusPainted(false);
        btn.setFocusable(false);
        btn.setBorder(new EmptyBorder(10, 15, 10, 15));
        btn.setCursor(new Cursor(Cursor.HAND_CURSOR));
        btn.setHorizontalAlignment(SwingConstants.LEFT);
        return btn;
    }

    private void addQuickChips() {
        // Đã chuyển sang Sidebar
    }

    private void selectFile() {
        FileDialog fileDialog = new FileDialog(this, "Chọn hồ sơ/CV", FileDialog.LOAD);
        fileDialog.setFile("*.pdf;*.docx"); // Gợi ý định dạng
        fileDialog.setVisible(true);
        
        String directory = fileDialog.getDirectory();
        String file = fileDialog.getFile();
        
        if (file != null) {
            selectedFilePath = new File(directory, file).getAbsolutePath();
            statusLabel.setText("● Chọn file: " + file);
            statusLabel.setForeground(COLOR_ACCENT);
        }
    }

    private void sendMessage() { sendMessage(null); }

    private void sendMessage(String forcedAgentId) {
        if (!isReady) return; 
        String reqText = inputField.getText().trim();
        if (reqText.isEmpty() && selectedFilePath == null) return;
        String displayQuery = reqText;
        if (selectedFilePath != null && reqText.isEmpty()) displayQuery = "[Phân tích] " + new File(selectedFilePath).getName();
        addChatMessage(displayQuery, true);
        inputField.setText(""); charCounterLabel.setText("0/500"); setLoading(true);

        String jsonInput;
        if (selectedFilePath != null) {
            jsonInput = String.format("{\"question\": \"%s\", \"file_path\": \"%s\"%s}", 
                reqText.replace("\"", "\\\""), 
                selectedFilePath.replace("\\", "\\\\").replace("\"", "\\\""),
                forcedAgentId != null ? ", \"agent_id\": \"" + forcedAgentId + "\"" : "");
        } else {
            jsonInput = String.format("{\"question\": \"%s\"%s}", 
                reqText.replace("\"", "\\\""),
                forcedAgentId != null ? ", \"agent_id\": \"" + forcedAgentId + "\"" : "");
        }
        selectedFilePath = null;
        new SwingWorker<String, Void>() {
            @Override protected String doInBackground() throws Exception { return callPythonBridge(jsonInput); }
            @Override protected void done() { try { processAIResponse(get()); } catch (Exception e) { addChatMessage("Lỗi kết nối", false); } setLoading(false); }
        }.execute();
    }

    private String callPythonBridge(String jsonInput) {
        StringBuilder sb = new StringBuilder();
        try {
            Process p = new ProcessBuilder(PYTHON_EXECUTABLE, BRIDGE_SCRIPT, jsonInput).redirectErrorStream(true).start();
            try (BufferedReader r = new BufferedReader(new InputStreamReader(p.getInputStream(), StandardCharsets.UTF_8))) {
                String l; while ((l = r.readLine()) != null) sb.append(l).append("\n");
            }
            p.waitFor(); return sb.toString().trim();
        } catch (Exception e) { return "{\"error\": \""+e.getMessage()+"\"}"; }
    }

    private void processAIResponse(String rawJson) {
        String answer = "Lỗi hệ thống", agent = "None";
        try {
            if (rawJson.contains("\"answer\":")) {
                int start = rawJson.indexOf("\"answer\":") + 10;
                int end = rawJson.indexOf("\", \"sources\":");
                if (end == -1) end = rawJson.lastIndexOf("\"");
                answer = rawJson.substring(start, end).replace("\\n", "\n").replace("\\\"", "\"").trim();
                while(answer.startsWith("\"")) answer = answer.substring(1);
                while(answer.endsWith("\"")) answer = answer.substring(0, answer.length()-1);
            }
            if (rawJson.contains("\"sources\":")) {
                int startSrc = rawJson.indexOf("\"sources\":") + 10;
                int endSrc = rawJson.indexOf("],", startSrc);
                if (endSrc == -1) endSrc = rawJson.indexOf("]", startSrc);
                if (startSrc != 9 && endSrc != -1) {
                    String srcArrayStr = rawJson.substring(startSrc, endSrc).trim();
                    if (srcArrayStr.startsWith("[")) srcArrayStr = srcArrayStr.substring(1);
                    if (srcArrayStr.endsWith("]")) srcArrayStr = srcArrayStr.substring(0, srcArrayStr.length() - 1);
                    
                    if (!srcArrayStr.isEmpty()) {
                        String[] srcItems = srcArrayStr.split("\",\\s*\"");
                        String sourcesHtml = "\n\n<br><b>Nguồn tham khảo:</b><br><ul>";
                        for (String s : srcItems) {
                            s = s.replace("\"", "").trim();
                            if (s.startsWith("[Web] ")) {
                                String url = s.substring(6).trim();
                                sourcesHtml += "<li><a href=\"" + url + "\">" + url + "</a></li>";
                            } else if (s.startsWith("[USER_CV] ")) {
                                String fname = s.substring(10).trim();
                                sourcesHtml += "<li><a href=\"file://" + fname + "\">" + fname + "</a></li>";
                            } else {
                                sourcesHtml += "<li><a href=\"file://" + s + "\">" + s + "</a></li>";
                            }
                        }
                        sourcesHtml += "</ul>";
                        
                        // Xoá phần "Nguồn:" text thô do LLM tự sinh (nếu có) để tránh lặp
                        int nguonIndex = answer.lastIndexOf("Nguồn");
                        if (nguonIndex == -1) nguonIndex = answer.lastIndexOf("NGUỒN");
                        if (nguonIndex != -1 && nguonIndex > answer.length() - 1500) {
                            int startIndex = nguonIndex;
                            while (startIndex > 0) {
                                char c = answer.charAt(startIndex - 1);
                                if (c == '*' || c == '#' || c == ' ' || c == '\n' || c == '\r' || c == '-') {
                                    startIndex--;
                                } else {
                                    break;
                                }
                            }
                            answer = answer.substring(0, startIndex).trim();
                        }
                        
                        answer += sourcesHtml;
                    }
                }
            }
            if (rawJson.contains("\"agent\":")) {
                int start = rawJson.indexOf("\"agent\":") + 9;
                int end = rawJson.indexOf("\",", start);
                if (end == -1) end = rawJson.indexOf("}", start);
                agent = rawJson.substring(start, end).replace("\"", "").trim();
            }
            agentLabel.setText("Mode: " + agent);
            addChatMessage(answer, false);
        } catch (Exception e) { addChatMessage("Lỗi parse phản hồi", false); }
    }

    private void addChatMessage(String text, boolean isUser) {
        if (text == null || text.trim().isEmpty()) return;
        JEditorPane pane = new JEditorPane();
        pane.setContentType("text/html");
        pane.setText(markdownToHtml(text, isUser));
        pane.setEditable(false);
        pane.setOpaque(false);
        pane.setBackground(new Color(0,0,0,0));
        pane.putClientProperty(JEditorPane.HONOR_DISPLAY_PROPERTIES, Boolean.TRUE);
        pane.setBorder(new EmptyBorder(12, 12, 12, 12)); // Sử dụng Border thay vì Margin để padding ổn định hơn
        
        // Giới hạn chiều rộng tối đa 650px
        pane.setSize(new Dimension(600, Integer.MAX_VALUE)); 
        int preferredHeight = pane.getPreferredSize().height;
        pane.setPreferredSize(new Dimension(Math.min(pane.getPreferredSize().width + 30, 630), preferredHeight + 10));
        
        pane.addHyperlinkListener(e -> {
            if (e.getEventType() == HyperlinkEvent.EventType.ACTIVATED) {
                try { Desktop.getDesktop().browse(e.getURL().toURI()); } catch (Exception ex) { ex.printStackTrace(); }
            }
        });

        JPanel bubble = new JPanel(new BorderLayout()) {
            @Override protected void paintComponent(Graphics g) {
                Graphics2D g2 = (Graphics2D) g.create();
                g2.setRenderingHint(RenderingHints.KEY_ANTIALIASING, RenderingHints.VALUE_ANTIALIAS_ON);
                g2.setColor(isUser ? COLOR_GPA_BLUE : COLOR_AI_BUBBLE);
                g2.fill(new RoundRectangle2D.Double(0, 0, getWidth()-1, getHeight()-1, 25, 25)); // Bo tròn hơn
                if (!isUser) {
                    g2.setColor(new Color(225, 230, 235));
                    g2.draw(new RoundRectangle2D.Double(0, 0, getWidth()-1, getHeight()-1, 25, 25));
                }
                g2.dispose();
            }
        };
        bubble.setOpaque(false);
        bubble.add(pane);

        // --- Layout mới: Tự co dãn (Shrink-wrap) ---
        JPanel row = new JPanel(new FlowLayout(isUser ? FlowLayout.RIGHT : FlowLayout.LEFT, 15, 5));
        row.setOpaque(false);
        row.setMaximumSize(new Dimension(2000, 1000));
        
        if (!isUser) {
            try {
                ImageIcon icon = new ImageIcon("logo.png");
                Image img = icon.getImage().getScaledInstance(35, 35, Image.SCALE_SMOOTH);
                JLabel avatar = new JLabel(new ImageIcon(img));
                row.add(avatar);
            } catch (Exception e) {}
            row.add(bubble);
        } else {
            row.add(bubble);
            JLabel userAvatar = new JLabel("Me") {
                @Override protected void paintComponent(Graphics g) {
                    Graphics2D g2 = (Graphics2D) g.create();
                    g2.setRenderingHint(RenderingHints.KEY_ANTIALIASING, RenderingHints.VALUE_ANTIALIAS_ON);
                    g2.setColor(COLOR_GPA_BLUE);
                    g2.fillOval(0, 0, getWidth()-1, getHeight()-1);
                    super.paintComponent(g2); g2.dispose();
                }
            };
            userAvatar.setPreferredSize(new Dimension(30, 30));
            userAvatar.setHorizontalAlignment(SwingConstants.CENTER);
            userAvatar.setForeground(Color.WHITE);
            row.add(userAvatar);
        }

        messagesWrapper.add(row);
        messagesWrapper.revalidate(); messagesWrapper.repaint();
        SwingUtilities.invokeLater(() -> {
            JScrollBar v = scrollPane.getVerticalScrollBar();
            v.setValue(v.getMaximum());
        });
    }

    private String markdownToHtml(String text) { return markdownToHtml(text, false); }
    
    private String markdownToHtml(String text, boolean isUser) {
        if (text == null) return "";
        String html = text;

        // 1. Xử lý escape HTML cơ bản
        html = html.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;");
        
        // 2. Markdown Link: [text](url) -> <a href='url'>text</a>
        Pattern linkPattern = Pattern.compile("\\[(.*?)\\]\\((.*?)\\)");
        Matcher matcher = linkPattern.matcher(html);
        html = matcher.replaceAll("<a href=\"$2\">$1</a>");

        // 3. Bold: **text** -> <b>text</b>
        html = html.replaceAll("\\*\\*(.*?)\\*\\*", "<b>$1</b>");

        // 4. Headers
        html = html.replaceAll("(?m)^### (.*)$", "<h3>$1</h3>");
        html = html.replaceAll("(?m)^## (.*)$", "<h2>$1</h2>");
        html = html.replaceAll("(?m)^# (.*)$", "<h1>$1</h1>");

        // 5. Bullet points
        html = html.replaceAll("(?m)^[\\*\\-] (.*)$", "<li>$1</li>");
        if (html.contains("<li>")) {
            // Bao bọc bằng ul nhưng tránh lồng nhiều lần
            html = html.replaceAll("(<li>.*</li>)+", "<ul>$0</ul>");
        }

        // 6. Xuống dòng
        html = html.replace("\n", "<br>");
        
        // 7. Khôi phục các thẻ HTML an toàn được sinh ra bởi hệ thống (như link click được, danh sách, in đậm...)
        html = html.replace("&lt;br&gt;", "<br>")
                   .replace("&lt;b&gt;", "<b>")
                   .replace("&lt;/b&gt;", "</b>")
                   .replace("&lt;ul&gt;", "<ul>")
                   .replace("&lt;/ul&gt;", "</ul>")
                   .replace("&lt;li&gt;", "<li>")
                   .replace("&lt;/li&gt;", "</li>")
                   .replaceAll("&lt;a href=\"([^\"]+)\"&gt;", "<a href=\"$1\">")
                   .replace("&lt;/a&gt;", "</a>");
        
        String textColor = isUser ? "#FFFFFF" : "#212529";
        String linkColor = isUser ? "#FFD34E" : "#1A4B9E";

        return "<html><head><style>" +
               "body { font-family: 'Segoe UI', Tahoma, sans-serif; font-size: 13px; color: " + textColor + "; padding: 5px; margin: 0; line-height: 1.4; }" +
               "h1, h2, h3 { color: " + (isUser ? "#FFFFFF" : "#1A4B9E") + "; margin-top: 12px; margin-bottom: 6px; }" +
               "h1 { font-size: 18px; } h2 { font-size: 16px; } h3 { font-size: 14px; }" +
               "a { color: " + linkColor + "; text-decoration: underline; font-weight: bold; }" +
               "ul { padding-left: 22px; margin-top: 5px; margin-bottom: 5px; }" +
               "li { margin-bottom: 4px; }" +
               "b { font-weight: bold; }" +
               "</style></head><body>" + html + "</body></html>";
    }

    private void setLoading(boolean loading) {
        sendButton.setEnabled(!loading); inputField.setEnabled(!loading);
        statusLabel.setText(loading ? "● Analyzing..." : "● System Ready");
        statusLabel.setForeground(loading ? COLOR_ACCENT : new Color(46, 204, 113));
    }

    private void resetChat() {
        messagesWrapper.removeAll();
        addChatMessage("Xin chào! Tôi là Trợ lý AI chuyên nghiệp của GPA. Hãy gửi CV hoặc đặt câu hỏi về hướng nghiệp cho tôi nhé!", false);
        messagesWrapper.revalidate();
        messagesWrapper.repaint();
        agentLabel.setText("Mode: Waiting...");
        statusLabel.setText("● System Ready");
        statusLabel.setForeground(new Color(46, 204, 113));
    }

    public static void main(String[] args) {
        System.setProperty("apple.awt.antialiasing", "true");
        SwingUtilities.invokeLater(() -> new HollandRAGGUI().setVisible(true));
    }
}
